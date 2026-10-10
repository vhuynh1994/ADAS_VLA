// vla_stream_server: ADAS-VLA perception on the SA8650P HTP for a PC client over TCP.
//
//   vla_stream_server --det yolo11s_w8a8.bin --lane yolop_lane_w8a16_o3v8.bin [--port 50052]
//                     [--backend libQnnHtp.so] [--system libQnnSystem.so] [--perf burst|default]
//
// Both context binaries are loaded once. Per frame the client sends one JPEG of the letterboxed frame (the input size
// of both models, 384x640 here); the server decodes it, quantizes it for each model through per-channel lookup tables
// the client computed from the tensor encodings, runs the detector and the lane model on the HTP, and returns the
// detector boxes after the same filtering as Ultralytics (best class over all classes > conf, class filter, per-class
// NMS) plus the lane mask (argmax of the 2 lane channels) run-length encoded. Tracking, distances, the safety gate
// and the HUD stay on the PC (adas_vla/perception/board.py).
//
// Protocol "VLASTREAM 1" (little-endian uint32 unless noted)
//   on connect, server -> client text: "VLASTREAM 1", then per model "MODEL det|lane <graph>" followed by
//       "IN|OUT <name> <dtype hex> <scale> <offset> <rank> <d0> ... <dn>" per tensor, then "END"
//   client -> server: <kind> <nbytes> <payload>
//     kind 0 quit
//     kind 3 setup: uint16 det_lut[3][256], uint16 lane_lut[3][256] (pixel -> quantized input, per channel),
//            float conf, float iou, uint32 max_det, uint8 class_mask[num_classes]
//     kind 2 JPEG (H x W RGB, the models' input size) | kind 4 raw RGB uint8 H*W*3
//     kind 5 raw RGB uint8 H*W*3, reply = raw outputs (det OUT order, then lane OUT order) for bit-exact checks
//   server -> client: <status> <recv_us> <prep_us> <det_us> <lane_us> <post_us> <nboxes> <nruns> <payload>
//     payload kind 2/4: nboxes x {x1 y1 x2 y2 score cls} float32 in model-input pixels, best first, then nruns uint32
//     run lengths of the flattened H x W lane mask, alternating background / lane, starting with background.
//     status 0 ok, 1 bad request, 2 JPEG error, else the QNN error code.
//
// Build on the PC with the QNX SDP 8.0: board/build_qnx.sh. Start / stop from the PC: board/board_server.sh.

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

namespace {

volatile sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

void logCallback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > QNN_LOG_LEVEL_WARN) return;
  fprintf(stderr, level == QNN_LOG_LEVEL_ERROR ? "[QNN ERROR] " : "[QNN WARN] ");
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

double nowUs() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Qnn_DataType_t encodes the bit width in its low byte as hex digits: 0x08, 0x16, 0x32, 0x64.
size_t dtypeBytes(Qnn_DataType_t t) {
  switch (static_cast<uint32_t>(t) & 0xFF) {
    case 0x08: return 1;
    case 0x16: return 2;
    case 0x32: return 4;
    case 0x64: return 8;
    default: return 0;
  }
}

#define TFIELD(t, f) ((t).version == QNN_TENSOR_VERSION_2 ? (t).v2.f : (t).v1.f)

struct Slot {
  Qnn_Tensor_t tensor;
  std::string name;
  std::vector<uint32_t> dims;
  std::vector<uint8_t> buf;
  size_t elem = 0;
  float scale = 0.f;
  int32_t offset = 0;

  // quantized element i as an integer (unsigned fixed point, 8 or 16 bit)
  uint32_t q(size_t i) const {
    return elem == 1 ? buf[i] : reinterpret_cast<const uint16_t*>(buf.data())[i];
  }
  float deq(size_t i) const { return (static_cast<float>(q(i)) + static_cast<float>(offset)) * scale; }
};

bool copySlot(const Qnn_Tensor_t& src, Slot& s) {
  s.tensor = src;
  s.name = TFIELD(src, name) ? TFIELD(src, name) : "";
  const uint32_t rank = TFIELD(src, rank);
  s.dims.assign(TFIELD(src, dimensions), TFIELD(src, dimensions) + rank);
  const Qnn_QuantizeParams_t& qp = TFIELD(src, quantizeParams);
  if (qp.encodingDefinition != QNN_DEFINITION_DEFINED ||
      qp.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    fprintf(stderr, "tensor %s: only per-tensor scale/offset quantized IO is supported\n", s.name.c_str());
    return false;
  }
  s.scale = qp.scaleOffsetEncoding.scale;
  s.offset = qp.scaleOffsetEncoding.offset;
  s.elem = dtypeBytes(TFIELD(src, dataType));
  if (s.elem != 1 && s.elem != 2) {
    fprintf(stderr, "tensor %s: unsupported data type 0x%x\n", s.name.c_str(), TFIELD(src, dataType));
    return false;
  }
  size_t n = s.elem;
  for (uint32_t d : s.dims) n *= d;
  s.buf.assign(n, 0);
  return true;
}

void bindSlot(Slot& s) {
  Qnn_Tensor_t& t = s.tensor;
  if (t.version == QNN_TENSOR_VERSION_2) {
    t.v2.name = s.name.c_str();
    t.v2.dimensions = s.dims.data();
    t.v2.memType = QNN_TENSORMEMTYPE_RAW;
    t.v2.clientBuf.data = s.buf.data();
    t.v2.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size());
    t.v2.isDynamicDimensions = nullptr;
    t.v2.sparseParams = QNN_SPARSE_PARAMS_INIT;
  } else {
    t.v1.name = s.name.c_str();
    t.v1.dimensions = s.dims.data();
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = s.buf.data();
    t.v1.clientBuf.dataSize = static_cast<uint32_t>(s.buf.size());
  }
}

bool readFile(const char* path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  out.resize(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  return static_cast<bool>(f.read(reinterpret_cast<char*>(out.data()), out.size()));
}

bool recvAll(int fd, void* p, size_t n) {
  auto* c = static_cast<uint8_t*>(p);
  while (n) {
    ssize_t r = recv(fd, c, n, 0);
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool sendAll(int fd, const void* p, size_t n) {
  auto* c = static_cast<const uint8_t*>(p);
  while (n) {
    ssize_t r = send(fd, c, n, 0);
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

// Same intent as qnn-net-run --perf_profile burst: DCVS off, max voltage corners, low RPC latency.
void setBurst(const QNN_INTERFACE_VER_TYPE& q) {
  if (!q.deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (q.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) {
    fprintf(stderr, "perf: no device infrastructure, staying at default\n");
    return;
  }
  auto* htp = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  QnnHtpDevice_PerfInfrastructure_t perf = htp->perfInfra;
  uint32_t id = 0;
  if (perf.createPowerConfigId(0, 0, &id) != QNN_SUCCESS) {
    fprintf(stderr, "perf: createPowerConfigId failed\n");
    return;
  }
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
  memset(&dcvs, 0, sizeof(dcvs));
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = id;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = 0;
  dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  dcvs.dcvsV3Config.setSleepLatency = 1;
  dcvs.dcvsV3Config.sleepLatency = 40;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  QnnHtpPerfInfrastructure_PowerConfig_t rpcLatency;
  memset(&rpcLatency, 0, sizeof(rpcLatency));
  rpcLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpcLatency.rpcControlLatencyConfig = 100;
  QnnHtpPerfInfrastructure_PowerConfig_t rpcPolling;
  memset(&rpcPolling, 0, sizeof(rpcPolling));
  rpcPolling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpcPolling.rpcPollingTimeConfig = 9999;
  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&dcvs, &rpcLatency, &rpcPolling, nullptr};
  if (perf.setPowerConfig(id, cfgs) != QNN_SUCCESS)
    fprintf(stderr, "perf: setPowerConfig failed, staying at default\n");
  else
    fprintf(stderr, "perf: burst (DCVS off, max voltage corners, RPC polling)\n");
}

struct Model {
  std::string role, graphName;
  Qnn_ContextHandle_t context = nullptr;
  Qnn_GraphHandle_t graph = nullptr;
  std::vector<Slot> ins, outs;
  std::vector<Qnn_Tensor_t> inT, outT;
  uint16_t lut[3][256];
  bool nchw = true;
  uint32_t H = 0, W = 0;

  Qnn_ErrorHandle_t execute(const QNN_INTERFACE_VER_TYPE& q) {
    return q.graphExecute(graph, inT.data(), inT.size(), outT.data(), outT.size(), nullptr, nullptr);
  }

  // RGB uint8 H x W x 3 -> quantized input through the per-channel lookup table.
  void fill(const uint8_t* rgb) {
    Slot& in = ins[0];
    const size_t px = static_cast<size_t>(H) * W;
    if (in.elem == 1) {
      uint8_t* d = in.buf.data();
      for (size_t i = 0; i < px; ++i)
        for (int c = 0; c < 3; ++c) d[nchw ? c * px + i : 3 * i + c] = static_cast<uint8_t>(lut[c][rgb[3 * i + c]]);
    } else {
      auto* d = reinterpret_cast<uint16_t*>(in.buf.data());
      for (size_t i = 0; i < px; ++i)
        for (int c = 0; c < 3; ++c) d[nchw ? c * px + i : 3 * i + c] = lut[c][rgb[3 * i + c]];
    }
  }
};

bool loadModel(const char* path, const char* role, const QNN_INTERFACE_VER_TYPE& q,
               const QNN_SYSTEM_INTERFACE_VER_TYPE& sq, Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
               Model& m) {
  std::vector<uint8_t> blob;
  if (!readFile(path, blob)) {
    fprintf(stderr, "cannot read %s\n", path);
    return false;
  }
  QnnSystemContext_Handle_t sys = nullptr;
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t infoSize = 0;
  if (sq.systemContextCreate(&sys) != QNN_SUCCESS ||
      sq.systemContextGetBinaryInfo(sys, blob.data(), blob.size(), &info, &infoSize) != QNN_SUCCESS) {
    fprintf(stderr, "%s: cannot read binary info\n", path);
    return false;
  }
  uint32_t numGraphs = 0;
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      numGraphs = info->contextBinaryInfoV1.numGraphs; graphs = info->contextBinaryInfoV1.graphs; break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      numGraphs = info->contextBinaryInfoV2.numGraphs; graphs = info->contextBinaryInfoV2.graphs; break;
    default:
      numGraphs = info->contextBinaryInfoV3.numGraphs; graphs = info->contextBinaryInfoV3.graphs; break;
  }
  if (numGraphs != 1) {
    fprintf(stderr, "%s: expected 1 graph, found %u\n", path, numGraphs);
    return false;
  }
  const QnnSystemContext_GraphInfoV1_t& g = graphs[0].graphInfoV1;  // V1/V2/V3 share name, inputs, outputs
  m.role = role;
  m.graphName = g.graphName;
  m.ins.resize(g.numGraphInputs);
  m.outs.resize(g.numGraphOutputs);
  for (uint32_t i = 0; i < g.numGraphInputs; ++i)
    if (!copySlot(g.graphInputs[i], m.ins[i])) return false;
  for (uint32_t i = 0; i < g.numGraphOutputs; ++i)
    if (!copySlot(g.graphOutputs[i], m.outs[i])) return false;
  sq.systemContextFree(sys);
  for (auto& s : m.ins) bindSlot(s);
  for (auto& s : m.outs) bindSlot(s);
  if (m.ins.size() != 1 || m.ins[0].dims.size() != 4) {
    fprintf(stderr, "%s: expected one 4-D image input\n", path);
    return false;
  }
  const auto& d = m.ins[0].dims;
  m.nchw = d[1] == 3;
  if (!m.nchw && d[3] != 3) {
    fprintf(stderr, "%s: input is neither NCHW nor NHWC RGB\n", path);
    return false;
  }
  m.H = m.nchw ? d[2] : d[1];
  m.W = m.nchw ? d[3] : d[2];
  for (int c = 0; c < 3; ++c)
    for (int v = 0; v < 256; ++v) m.lut[c][v] = static_cast<uint16_t>(v);
  if (q.contextCreateFromBinary(backend, device, nullptr, blob.data(), blob.size(), &m.context, nullptr) != QNN_SUCCESS ||
      q.graphRetrieve(m.context, m.graphName.c_str(), &m.graph) != QNN_SUCCESS) {
    fprintf(stderr, "%s: context/graph creation failed\n", path);
    return false;
  }
  for (auto& s : m.ins) m.inT.push_back(s.tensor);
  for (auto& s : m.outs) m.outT.push_back(s.tensor);
  return true;
}

const Slot* findOut(const Model& m, const char* name) {
  for (auto& s : m.outs)
    if (s.name == name) return &s;
  return nullptr;
}

struct Box { float x1, y1, x2, y2, score, cls; };

// Ultralytics non_max_suppression (multi_label=False): candidates whose best class over ALL classes scores above conf,
// then the class filter, then NMS per class (boxes offset by class), best first, at most max_det.
struct DetDecoder {
  const Slot* scores = nullptr;  // [1, C, N]
  const Slot* boxes = nullptr;   // [1, 4, N] xywh in input pixels
  float conf = 0.35f, iou = 0.7f;
  uint32_t maxDet = 300;
  std::vector<uint8_t> classMask;

  void run(std::vector<Box>& out) const {
    out.clear();
    const uint32_t C = scores->dims[1], N = scores->dims[2];
    std::vector<Box> cand;
    for (uint32_t a = 0; a < N; ++a) {
      uint32_t best = 0, bq = scores->q(a);
      for (uint32_t c = 1; c < C; ++c) {
        const uint32_t v = scores->q(static_cast<size_t>(c) * N + a);
        if (v > bq) { bq = v; best = c; }  // first maximum, like torch.max
      }
      const float s = (static_cast<float>(bq) + static_cast<float>(scores->offset)) * scores->scale;
      if (!(s > conf) || best >= classMask.size() || !classMask[best]) continue;
      const float cx = boxes->deq(a), cy = boxes->deq(N + a), w = boxes->deq(2 * N + a), h = boxes->deq(3 * N + a);
      cand.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s, static_cast<float>(best)});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Box& x, const Box& y) { return x.score > y.score; });
    std::vector<bool> removed(cand.size(), false);
    for (size_t i = 0; i < cand.size() && out.size() < maxDet; ++i) {
      if (removed[i]) continue;
      const Box& b = cand[i];
      out.push_back(b);
      const float ab = (b.x2 - b.x1) * (b.y2 - b.y1);
      for (size_t j = i + 1; j < cand.size(); ++j) {
        const Box& o = cand[j];
        if (removed[j] || o.cls != b.cls) continue;
        const float iw = std::min(b.x2, o.x2) - std::max(b.x1, o.x1), ih = std::min(b.y2, o.y2) - std::max(b.y1, o.y1);
        if (iw <= 0 || ih <= 0) continue;
        const float inter = iw * ih, ao = (o.x2 - o.x1) * (o.y2 - o.y1);
        if (inter / (ab + ao - inter) > iou) removed[j] = true;
      }
    }
  }
};

// Lane mask = argmax over the 2 channels of [1, 2, H, W] (index 0 on ties, like numpy/torch argmax), run-length encoded.
void laneRuns(const Slot& seg, std::vector<uint32_t>& runs) {
  runs.clear();
  const size_t plane = static_cast<size_t>(seg.dims[2]) * seg.dims[3];
  bool cur = false;
  uint32_t len = 0;
  for (size_t i = 0; i < plane; ++i) {
    const bool lane = seg.q(plane + i) > seg.q(i);
    if (lane != cur) {
      runs.push_back(len);
      len = 0;
      cur = lane;
    }
    ++len;
  }
  runs.push_back(len);
}

}  // namespace

int main(int argc, char** argv) {
  std::string detPath, lanePath, backendLib = "libQnnHtp.so", systemLib = "libQnnSystem.so", perfMode = "burst";
  int port = 50052;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--det") detPath = argv[i + 1];
    else if (k == "--lane") lanePath = argv[i + 1];
    else if (k == "--port") port = atoi(argv[i + 1]);
    else if (k == "--backend") backendLib = argv[i + 1];
    else if (k == "--system") systemLib = argv[i + 1];
    else if (k == "--perf") perfMode = argv[i + 1];
  }
  if (detPath.empty() || lanePath.empty()) {
    fprintf(stderr, "usage: %s --det det.bin --lane lane.bin [--port N] [--backend lib] [--system lib] "
            "[--perf burst|default]\n", argv[0]);
    return 2;
  }

  void* hBackend = dlopen(backendLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  void* hSystem = dlopen(systemLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!hBackend || !hSystem) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
  auto getProviders = reinterpret_cast<GetProviders>(dlsym(hBackend, "QnnInterface_getProviders"));
  auto getSysProviders = reinterpret_cast<GetSysProviders>(dlsym(hSystem, "QnnSystemInterface_getProviders"));
  const QnnInterface_t** providers = nullptr;
  const QnnSystemInterface_t** sysProviders = nullptr;
  uint32_t nProv = 0, nSys = 0;
  if (!getProviders || !getSysProviders || getProviders(&providers, &nProv) != QNN_SUCCESS ||
      getSysProviders(&sysProviders, &nSys) != QNN_SUCCESS || nProv == 0 || nSys == 0) {
    fprintf(stderr, "cannot get QNN providers\n");
    return 1;
  }
  const QNN_INTERFACE_VER_TYPE* q = nullptr;
  for (uint32_t i = 0; i < nProv && !q; ++i)
    if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR)
      q = &providers[i]->QNN_INTERFACE_VER_NAME;
  const QNN_SYSTEM_INTERFACE_VER_TYPE* sq = &sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  if (!q) {
    fprintf(stderr, "no provider for QNN API major %d\n", QNN_API_VERSION_MAJOR);
    return 1;
  }

  double t0 = nowUs();
  Qnn_LogHandle_t log = nullptr;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  q->logCreate(logCallback, QNN_LOG_LEVEL_WARN, &log);
  if (q->backendCreate(log, nullptr, &backend) != QNN_SUCCESS) {
    fprintf(stderr, "backendCreate failed\n");
    return 1;
  }
  if (q->deviceCreate && q->deviceCreate(log, nullptr, &device) != QNN_SUCCESS) {
    fprintf(stderr, "deviceCreate failed (CDSP_LIBRARY_PATH set?)\n");
    return 1;
  }
  if (perfMode == "burst") setBurst(*q);

  Model det, lane;
  if (!loadModel(detPath.c_str(), "det", *q, *sq, backend, device, det) ||
      !loadModel(lanePath.c_str(), "lane", *q, *sq, backend, device, lane))
    return 1;
  if (det.H != lane.H || det.W != lane.W) {
    fprintf(stderr, "det input %ux%u != lane input %ux%u: both models must take the same frame\n", det.H, det.W,
            lane.H, lane.W);
    return 1;
  }
  DetDecoder dec;
  dec.scores = findOut(det, "scores");
  dec.boxes = findOut(det, "boxes");
  const Slot* seg = findOut(lane, "lane_line_seg");
  if (!dec.scores || !dec.boxes || !seg || seg->dims.size() != 4 || seg->dims[1] != 2 || seg->dims[2] != lane.H ||
      seg->dims[3] != lane.W) {
    fprintf(stderr, "unexpected outputs: need det scores [1,C,N] + boxes [1,4,N], lane lane_line_seg [1,2,H,W]\n");
    return 1;
  }
  const uint32_t numClasses = dec.scores->dims[1];
  dec.classMask.assign(numClasses, 1);
  det.execute(*q);  // warm-up so the first client frame does not pay for it
  lane.execute(*q);
  fprintf(stderr, "ready: det %s + lane %s on %ux%u, init %.0f ms\n", det.graphName.c_str(), lane.graphName.c_str(),
          det.W, det.H, (nowUs() - t0) / 1e3);

  std::string header = "VLASTREAM 1\n";
  auto describe = [&header](const char* kind, const Slot& s) {
    char line[512];
    int n = snprintf(line, sizeof(line), "%s %s 0x%x %.9g %d %zu", kind, s.name.c_str(),
                     static_cast<unsigned>(TFIELD(s.tensor, dataType)), s.scale, s.offset, s.dims.size());
    header.append(line, static_cast<size_t>(n));
    for (uint32_t d : s.dims) header += " " + std::to_string(d);
    header += "\n";
  };
  for (Model* m : {&det, &lane}) {
    header += "MODEL " + m->role + " " + m->graphName + "\n";
    for (auto& s : m->ins) describe("IN", s);
    for (auto& s : m->outs) describe("OUT", s);
  }
  header += "END\n";

  struct sigaction sa {};
  sa.sa_handler = onSignal;  // no SA_RESTART: a signal must break out of the blocking accept()/recv()
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(ls, 1) != 0) {
    fprintf(stderr, "cannot listen on port %d: %s\n", port, strerror(errno));
    return 1;
  }
  fprintf(stderr, "listening on port %d (Ctrl-C to stop)\n", port);

  const size_t lutBytes = 2 * 3 * 256 * sizeof(uint16_t);
  std::vector<uint8_t> msg, reply;
  std::vector<Box> boxes;
  std::vector<uint32_t> runs;
  while (!g_stop) {
    int fd = accept(ls, nullptr, nullptr);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    fprintf(stderr, "client connected\n");
    sendAll(fd, header.data(), header.size());
    uint64_t frames = 0;
    double detSum = 0, laneSum = 0, prepSum = 0;
    bool setup = false;
    while (!g_stop) {
      uint32_t hdr[2];
      if (!recvAll(fd, hdr, 8) || hdr[0] == 0) break;
      const uint32_t kind = hdr[0], n = hdr[1];
      double r0 = nowUs();
      msg.resize(n);
      if (n && !recvAll(fd, msg.data(), n)) break;
      double r1 = nowUs(), p1 = r1, e1 = r1, e2 = r1, d1 = r1;
      uint32_t status = 0;
      boxes.clear();
      runs.clear();
      reply.assign(32, 0);

      if (kind == 3) {
        if (n != lutBytes + 12 + numClasses) {
          status = 1;
        } else {
          memcpy(det.lut, msg.data(), sizeof(det.lut));
          memcpy(lane.lut, msg.data() + sizeof(det.lut), sizeof(lane.lut));
          memcpy(&dec.conf, msg.data() + lutBytes, 4);
          memcpy(&dec.iou, msg.data() + lutBytes + 4, 4);
          memcpy(&dec.maxDet, msg.data() + lutBytes + 8, 4);
          memcpy(dec.classMask.data(), msg.data() + lutBytes + 12, numClasses);
          setup = true;
        }
      } else if (kind == 2 || kind == 4 || kind == 5) {
        const uint8_t* rgb = nullptr;
        stbi_uc* decoded = nullptr;
        if (!setup) {
          status = 1;
        } else if (kind == 2) {
          int w = 0, h = 0, comp = 0;
          decoded = stbi_load_from_memory(msg.data(), static_cast<int>(n), &w, &h, &comp, 3);
          if (!decoded || w != static_cast<int>(det.W) || h != static_cast<int>(det.H)) status = 2;
          rgb = decoded;
        } else if (n != det.W * det.H * 3) {
          status = 1;
        } else {
          rgb = msg.data();
        }
        if (status == 0) {
          det.fill(rgb);
          lane.fill(rgb);
        }
        if (decoded) stbi_image_free(decoded);
        p1 = nowUs();
        if (status == 0) status = static_cast<uint32_t>(det.execute(*q));
        e1 = nowUs();
        if (status == 0) status = static_cast<uint32_t>(lane.execute(*q));
        e2 = nowUs();
        if (status == 0 && kind == 5) {
          for (Model* m : {&det, &lane})
            for (auto& s : m->outs) reply.insert(reply.end(), s.buf.begin(), s.buf.end());
        } else if (status == 0) {
          dec.run(boxes);
          laneRuns(*seg, runs);
          const auto* b = reinterpret_cast<const uint8_t*>(boxes.data());
          reply.insert(reply.end(), b, b + boxes.size() * sizeof(Box));
          const auto* r = reinterpret_cast<const uint8_t*>(runs.data());
          reply.insert(reply.end(), r, r + runs.size() * sizeof(uint32_t));
        }
        d1 = nowUs();
        if (status == 0) {
          ++frames;
          prepSum += p1 - r1;
          detSum += e1 - p1;
          laneSum += e2 - e1;
        }
      } else {
        status = 1;
      }
      uint32_t rh[8] = {status, static_cast<uint32_t>(r1 - r0), static_cast<uint32_t>(p1 - r1),
                        static_cast<uint32_t>(e1 - p1), static_cast<uint32_t>(e2 - e1), static_cast<uint32_t>(d1 - e2),
                        static_cast<uint32_t>(boxes.size()), static_cast<uint32_t>(runs.size())};
      memcpy(reply.data(), rh, sizeof(rh));
      if (!sendAll(fd, reply.data(), reply.size())) break;
    }
    close(fd);
    fprintf(stderr, "client gone after %llu frames: mean jpeg+quantize %.2f ms, det %.2f ms, lane %.2f ms\n",
            static_cast<unsigned long long>(frames), frames ? prepSum / frames / 1e3 : 0.0,
            frames ? detSum / frames / 1e3 : 0.0, frames ? laneSum / frames / 1e3 : 0.0);
  }
  close(ls);
  q->contextFree(det.context, nullptr);
  q->contextFree(lane.context, nullptr);
  if (device && q->deviceFree) q->deviceFree(device);
  q->backendFree(backend);
  if (log) q->logFree(log);
  fprintf(stderr, "stopped\n");
  return 0;
}
