// VLM server for the SA8650P (QNX 8.0): keeps a Genie VLM pipeline loaded and answers decision requests from the PC
// pipeline (adas_vla/reasoning/board_vlm.py) over TCP. Models: InternVL3.5-2B (Qualcomm AI Hub bundle) or, with
// --qwen HxW, a Qwen2.5-VL bundle built by scripts/v3_qnn.py (the fine-tuned adas-vlm-v3).
//
//   vlm_genie_server [--port 50053] [--dir /data/adas_vla_genie/internvl3_5_2b]
//                    [--image img-enc-nsp0.json] [--text text-encoder.json] [--gen greedy-text-generator-nsp1.json]
//                    [--qwen 308x560]
//
// Request : "VLM1", u32 len + prompt text before the image, u32 len + prompt text after it, u32 len + pixels
//           (float32 NCHW 1x3x448x448, ImageNet-normalised RGB), u32 max_new_tokens (0 = no limit).
//           "VLM2": the same with uint8 RGB HWC 448x448x3 pixels, normalised here with the same float32 operations
//           (4x fewer bytes: the time outside the server per request went from ~0.23 s to ~0.08 s).
//           "VLM3" (--qwen HxW): uint8 RGB HWC HxWx3, cut here into 14x14 patches in the Qwen2-VL processor layout
//           (float32 [H/14*W/14, 1176], CLIP normalisation as adas_vla/deploy/qwen_vl.py: patchify); the ViT's other
//           inputs (rotary cos/sin, window / full attention masks: vit_*.raw in --dir) are constant for the size.
// Response: "VLR1", i32 status (0 = ok), f32 image ms, f32 first token ms, f32 total ms, u32 tokens, u32 len + text.
// The image is set after the first prompt part and before the second, so the embeddings reach the LLM in prompt
// order. The text callback returns an error after max_new_tokens to ask Genie to stop, but the Genie 1.18 pipeline
// ignores it and generates the whole answer (~2.5 s): the action itself is ready after ~0.85 s.
// Run with LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so and the libQnnHtp.so shim first in LD_LIBRARY_PATH (README).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "GenieCommon.h"
#include "GenieNode.h"
#include "GeniePipeline.h"

using Clock = std::chrono::steady_clock;

struct Turn {  // state of one request, shared with the text callback through userData
  std::string text;
  Clock::time_point start, image_done, first;
  bool got_first = false;
  uint32_t tokens = 0, max_tokens = 0;
};

static Genie_Status_t onText(const char* response, const GenieNode_TextOutput_SentenceCode_t code, const void* user) {
  Turn* t = const_cast<Turn*>(static_cast<const Turn*>(user));
  if (response && *response) {
    if (!t->got_first) {
      t->first = Clock::now();
      t->got_first = true;
    }
    t->text += response;
    ++t->tokens;
  }
  (void)code;
  // A non-success status asks Genie to stop generating (used for max_new_tokens).
  return (t->max_tokens && t->tokens >= t->max_tokens) ? GENIE_STATUS_ERROR_GENERAL : GENIE_STATUS_SUCCESS;
}

static std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static bool readAll(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n) {
    ssize_t r = recv(fd, p, n, 0);
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

static bool writeAll(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n) {
    ssize_t w = send(fd, p, n, 0);
    if (w <= 0) return false;
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

static bool readBlob(int fd, std::string& out, size_t maxLen) {
  uint32_t len = 0;
  if (!readAll(fd, &len, 4) || len > maxLen) return false;
  out.resize(len);
  return len == 0 || readAll(fd, &out[0], len);
}

// uint8 RGB HWC -> float32 NCHW with the ImageNet normalisation, the same float32 operations as
// adas_vla/reasoning/board_vlm.py: pixel_values() ((x / 255 - mean) / std).
static void normalise(const std::string& rgb, std::string& out, int hw) {
  static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
  out.resize(static_cast<size_t>(3) * hw * sizeof(float));
  const unsigned char* src = reinterpret_cast<const unsigned char*>(rgb.data());
  float* dst = reinterpret_cast<float*>(&out[0]);
  for (int c = 0; c < 3; ++c)
    for (int i = 0; i < hw; ++i) dst[c * hw + i] = (src[i * 3 + c] / 255.0f - mean[c]) / stdv[c];
}

// uint8 RGB HWC (h x w) -> float32 [patches, 3 * 2 * 14 * 14] in the Qwen2-VL image processor layout: patches by 2x2
// merge unit, each row (channel, frame, y, x) with the single image used for both frames of the temporal patch;
// the same float32 operations as qwen_vl.patchify ((x * (1 / 255) - mean) / std).
static void patchifyQwen(const std::string& rgb, std::string& out, int h, int w) {
  static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f}, stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
  const int p = 14, gh = h / p, gw = w / p, row = 3 * 2 * p * p;
  out.resize(static_cast<size_t>(gh) * gw * row * sizeof(float));
  const unsigned char* src = reinterpret_cast<const unsigned char*>(rgb.data());
  float* dst = reinterpret_cast<float*>(&out[0]);
  int idx = 0;
  for (int bh = 0; bh < gh / 2; ++bh)
    for (int bw = 0; bw < gw / 2; ++bw)
      for (int mh = 0; mh < 2; ++mh)
        for (int mw = 0; mw < 2; ++mw, ++idx) {
          const int r = bh * 2 + mh, c = bw * 2 + mw;
          float* o = dst + static_cast<size_t>(idx) * row;
          for (int ch = 0; ch < 3; ++ch)
            for (int y = 0; y < p; ++y)
              for (int x = 0; x < p; ++x) {
                const float v = (src[((r * p + y) * w + c * p + x) * 3 + ch] * (1.0f / 255.0f) - mean[ch]) / stdv[ch];
                o[((ch * 2 + 0) * p + y) * p + x] = v;
                o[((ch * 2 + 1) * p + y) * p + x] = v;
              }
        }
}

static float ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<float, std::milli>(b - a).count();
}

#define CHECK(call)                                                       \
  do {                                                                    \
    Genie_Status_t s_ = (call);                                           \
    if (s_ != GENIE_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(s_));    \
      return 1;                                                           \
    }                                                                     \
  } while (0)

int main(int argc, char** argv) {
  int port = 50053;
  std::string dir = "/data/adas_vla_genie/internvl3_5_2b", imageCfg = "img-enc-nsp0.json",
              textCfg = "text-encoder.json", genCfg = "greedy-text-generator-nsp1.json";
  int qwenH = 0, qwenW = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--port") port = atoi(argv[i + 1]);
    else if (k == "--dir") dir = argv[i + 1];
    else if (k == "--image") imageCfg = argv[i + 1];
    else if (k == "--text") textCfg = argv[i + 1];
    else if (k == "--gen") genCfg = argv[i + 1];
    else if (k == "--qwen") sscanf(argv[i + 1], "%dx%d", &qwenH, &qwenW);
  }
  if (chdir(dir.c_str()) != 0) {
    fprintf(stderr, "cannot cd to %s\n", dir.c_str());
    return 1;
  }
  // Qwen2.5-VL ViT inputs that depend only on the image size (written by scripts/v3_qnn.py bundle)
  struct VitInput {
    GenieNode_IOName_t io;
    const char* file;
    std::string data;
  } vitInputs[] = {{GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS, "vit_position_ids_cos.raw", {}},
                   {GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN, "vit_position_ids_sin.raw", {}},
                   {GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK, "vit_window_attention_mask.raw", {}},
                   {GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK, "vit_full_attention_mask.raw", {}}};
  if (qwenH) {
    for (auto& v : vitInputs) {
      std::ifstream f(v.file, std::ios::binary);
      v.data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      if (v.data.empty()) {
        fprintf(stderr, "cannot read %s\n", v.file);
        return 1;
      }
    }
    fprintf(stderr, "Qwen2.5-VL mode: image %dx%d, %d patches\n", qwenH, qwenW, (qwenH / 14) * (qwenW / 14));
  }

  auto t0 = Clock::now();
  GenieNodeConfig_Handle_t imageConfig = nullptr, textConfig = nullptr, genConfig = nullptr;
  GenieNode_Handle_t image = nullptr, text = nullptr, gen = nullptr;
  GeniePipelineConfig_Handle_t pipeConfig = nullptr;
  GeniePipeline_Handle_t pipe = nullptr;
  CHECK(GenieNodeConfig_createFromJson(readFile(imageCfg).c_str(), &imageConfig));
  CHECK(GenieNode_create(imageConfig, &image));
  CHECK(GenieNodeConfig_createFromJson(readFile(textCfg).c_str(), &textConfig));
  CHECK(GenieNode_create(textConfig, &text));
  CHECK(GenieNodeConfig_createFromJson(readFile(genCfg).c_str(), &genConfig));
  CHECK(GenieNode_create(genConfig, &gen));
  CHECK(GenieNode_setTextCallback(gen, GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT, onText));
  CHECK(GeniePipelineConfig_createFromJson("", &pipeConfig));
  CHECK(GeniePipeline_create(pipeConfig, &pipe));
  CHECK(GeniePipeline_addNode(pipe, image));
  CHECK(GeniePipeline_addNode(pipe, text));
  CHECK(GeniePipeline_addNode(pipe, gen));
  CHECK(GeniePipeline_connect(pipe, image, GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT, gen,
                              GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT));
  CHECK(GeniePipeline_connect(pipe, text, GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT, gen,
                              GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT));
  fprintf(stderr, "VLM loaded in %.1f s\n", ms(t0, Clock::now()) / 1000.0f);

  int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(srv, 1) != 0) {
    fprintf(stderr, "cannot listen on port %d\n", port);
    return 1;
  }
  fprintf(stderr, "listening on port %d\n", port);
  const int hw = 448 * 448;
  const size_t pixelBytes = 3 * hw * sizeof(float), rgbBytes = 3 * hw;
  for (;;) {
    int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fprintf(stderr, "client connected\n");
    std::string before, after, pixels, rgb;
    for (;;) {
      char magic[4];
      uint32_t maxTokens = 0;
      if (!readAll(fd, magic, 4)) break;
      const bool v2 = memcmp(magic, "VLM2", 4) == 0, v3 = memcmp(magic, "VLM3", 4) == 0;
      if (!v2 && !v3 && memcmp(magic, "VLM1", 4) != 0) break;
      if (v3 != (qwenH > 0)) {
        fprintf(stderr, "request %.4s does not match the server mode (--qwen)\n", magic);
        break;
      }
      if (!readBlob(fd, before, 1 << 20) || !readBlob(fd, after, 1 << 20)) break;
      const size_t qwenBytes = static_cast<size_t>(qwenH) * qwenW * 3;
      if (v3 ? !readBlob(fd, rgb, qwenBytes) || rgb.size() != qwenBytes
          : v2 ? !readBlob(fd, rgb, rgbBytes) || rgb.size() != rgbBytes
               : !readBlob(fd, pixels, pixelBytes) || pixels.size() != pixelBytes)
        break;
      if (!readAll(fd, &maxTokens, 4)) break;
      if (v2) normalise(rgb, pixels, hw);
      Turn turn;
      turn.max_tokens = maxTokens;
      turn.start = Clock::now();
      if (v3) patchifyQwen(rgb, pixels, qwenH, qwenW);  // counted in the image time (preprocessing on the board)
      int32_t status = 0;
      if (GenieNode_setData(text, GENIE_NODE_TEXT_ENCODER_TEXT_INPUT, before.data(), before.size(), nullptr))
        status = 1;
      for (auto& v : vitInputs)  // Qwen: the constant ViT inputs go in with every image
        if (!status && qwenH && GenieNode_setData(image, v.io, v.data.data(), v.data.size(), nullptr)) status = 4;
      if (!status && GenieNode_setData(image, GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT, pixels.data(), pixels.size(),
                                       nullptr))
        status = 1;
      turn.image_done = Clock::now();
      if (!status && GenieNode_setData(text, GENIE_NODE_TEXT_ENCODER_TEXT_INPUT, after.data(), after.size(), nullptr))
        status = 2;
      if (!status) {
        Genie_Status_t s = GeniePipeline_execute(pipe, &turn);
        if (s != GENIE_STATUS_SUCCESS && !(turn.max_tokens && turn.tokens >= turn.max_tokens)) status = 3;
      }
      GeniePipeline_reset(pipe);
      Clock::time_point end = Clock::now();
      float timing[3] = {ms(turn.start, turn.image_done), turn.got_first ? ms(turn.start, turn.first) : -1.0f,
                         ms(turn.start, end)};
      uint32_t len = static_cast<uint32_t>(turn.text.size());
      if (!writeAll(fd, "VLR1", 4) || !writeAll(fd, &status, 4) || !writeAll(fd, timing, sizeof(timing)) ||
          !writeAll(fd, &turn.tokens, 4) || !writeAll(fd, &len, 4) || !writeAll(fd, turn.text.data(), len))
        break;
    }
    close(fd);
    fprintf(stderr, "client disconnected\n");
  }
}
