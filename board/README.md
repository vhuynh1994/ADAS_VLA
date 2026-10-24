# Perception trên board SA8650P, hiển thị trên PC

`vla_stream_server` chạy trên board (QNX 8.0, HTP v73), nạp **một lần** 2 context binary: detector YOLO11s W8A8 và
lane YOLOP 384×640 W8A16 (`scripts/make_yolop_lane.py`). PC gửi từng frame qua TCP; board chạy cả hai model rồi trả về
box và mask lane; PC chạy phần còn lại của pipeline như bình thường (ByteTrack, fit lane, khoảng cách/TTC, safety gate,
controller, HUD, và VLM nếu bật).

```
PC: frame → letterbox 384×640 → JPEG q90 ──TCP──►
Board: giải JPEG → lượng tử (bảng tra theo encoding của từng model) → HTP: YOLO11s → HTP: lane
       → lọc box + NMS (như Ultralytics) + mask lane (argmax, RLE) ──TCP──►
PC: ByteTrack → fit lane (Hough) → geometry / TTC → safety gate → controller → HUD (cửa sổ / mp4 / log)
```

## Chạy

```bash
export QNN_SDK_ROOT=<QAIRT 2.46 SDK root>                # có include/QNN
bash board/build_qnx.sh                                 # biên dịch cho QNX 8.0 (QNX SDP ~/qnx800)
bash board/board_server.sh deploy                       # copy server vào $DIR/bin (mặc định /data/adas_vla_perf)
bash board/board_server.sh start                        # nạp model, giữ HTP ở chế độ burst
adas-vla run --no-vlm --set perception.backend=board --source data/samples/highway_traffic.mp4 --show
bash board/board_server.sh stop                         # nhớ dừng: board dùng chung, server giữ HTP ở burst
```

Xem trực tiếp: `bash board/demo.sh <video>` mở cửa sổ (qua `systemd-run`, unit `adas-vla-demo`); dừng bằng
`q` / Esc / đóng cửa sổ, hoặc `bash board/demo.sh stop`. HUD vẽ trên frame thu về `--display-width` (mặc định 1280),
nên video 4K vẫn vừa màn hình; `--display-width 0` = kích thước gốc.

`$DIR` trên board phải có `lib/` (libQnnHtp.so, libQnnHtpV73Stub.so, libQnnSystem.so cho aarch64-qnx800) và
`models/` (`yolo11s_w8a16_p5fp16_o3v8.bin` hoặc bản cũ `yolo11s_w8a8.bin`, `yolop_lane_w8a16_o3v8.bin`); đổi bằng biến `DIR`, `DET`, `LANE`, `PORT` (mặc định
50052). Từ terminal VS Code (snap) mở cửa sổ qua
`systemd-run --user --collect -p WorkingDirectory=$PWD $PWD/.venv/bin/adas-vla run ... --show`.
Có thể bật VLM trên GPU của PC cùng lúc: bỏ `--no-vlm`, thêm `--vlm-mode process`.

## Đã kiểm chứng (2026-10-10)

- Output của 2 model qua server (bảng tra lượng tử) **trùng từng bit** với `qnn-net-run` trên 20 ảnh mỗi model.
  Bảng tra phải dùng đúng công thức `datautil::floatToTfN` của SDK (`encodingMin = offset * scale` tính bằng float,
  làm tròn kiểu C) trên giá trị chuẩn hoá float32; `round(x / scale) - offset` lệch 1 ở ~0.3 % phần tử input 16-bit,
  đủ để output khác ở ~5000 phần tử.
- Mask lane trên board trùng argmax trên PC (20/20).
- Box: trùng `non_max_suppression` của Ultralytics trừ khi nhiều anchor có cùng điểm (điểm 8-bit, bước 0.0037): board
  giữ anchor đầu tiên, torchvision không cố định thứ tự, nên box giữ lại lệch nhau một bước lượng tử (~2.6 px).

## Hiệu năng (`highway_traffic.mp4` 1280×720, 600 frame, p50, ms)

| Bước | ms |
|---|---|
| PC: letterbox + JPEG | 0.9 |
| Mạng (JPEG ~40 KB lên, box + RLE về) | 5.3 |
| Board: giải JPEG + lượng tử cho 2 model | 5.6 |
| **Board: HTP YOLO11s** | **2.4** |
| **Board: HTP lane 384×640** | **4.7** |
| Board: lọc box + NMS + RLE | 1.0 |
| PC: ByteTrack + tạo Detection | 1.2 |
| PC: fit lane (Hough) | 9.0 |
| **Capture → lệnh điều khiển** | **31.0** (p95 37.1) |

Chạy cả HUD + ghi video: 21.6 FPS. Trên xe thật camera nối thẳng vào board nên không có chặng mạng và JPEG; phần
còn lại đáng tối ưu là fit lane trên CPU và giải JPEG + lượng tử trên board (một luồng).

## Detector W8A16 + đường stride 32 FP16 (12/10/2026, mặc định từ nay)

Bản W8A8 cũ bỏ sót vật lớn (xe gần) và sinh box thừa; W8A16 thuần thì hỏng từ layer 8 (phép nhân SiLU 16-bit ở stride 32,
không phải attention C2PSA). Giữ đường stride 32 ở FP16 (`scripts/qnn_fp16_overrides.py`), phần còn lại W8A16, O3 + VTCM 8:

| YOLO11s trên HTP NSP0 | latency p50 | So với float, 20 ảnh (vật nhỏ / vừa / lớn, box thừa) | Vật/frame trên highway_traffic (float PC: 1.39) |
|---|---|---|---|
| W8A8 (cũ, cấu hình mặc định) | 2.16 ms | 73 % / 96 % / 76 %, 28 box thừa | 2.02 |
| W8A8, O3 + VTCM 8 | 2.03 ms | như trên | – |
| **W8A16 + P5 FP16, O3 + VTCM 8** | **4.88 ms** | **98 % / 100 % / 100 %, 0 box thừa** (đo trên board) | **1.30** |

Output trên board lệch nhẹ so với emulator (score lệch ≤ 0.016, do phép FP16 trên chip) nhưng kết quả phát hiện giống hệt.
Box là uint16 với bước 0.01 px (W8A8: 2.58 px), nên khoảng cách theo chiều cao box ít bị nhiễu lượng tử hơn. Cả pipeline
trên board: 33.5 ms p50 (W8A8: 31.0 ms), 21.7 FPS kể cả HUD.

## VLM trên board: InternVL3.5-2B (13/10/2026)

Bundle Genie W4A16 của Qualcomm AI Hub cho SA8775P (cùng HTP v73 / soc 52), không cần biên dịch lại:
`qai-hub-models fetch intern3_5_vl_2b -r genie -p w4a16 -c qualcomm-sa8775p` (2.4 GB), giải nén vào
`/data/adas_vla_genie/internvl3_5_2b`; `genie-app` lấy từ `bin/aarch64-qnx800` của SDK QAIRT 2.46 vào `/data/adas_vla_genie/bin`.
Bốn chỗ phải sửa để Genie chạy trên QNX 8.0 của board:

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `Unable to access symbols in libcdsprpc` | QNX để `rpcmem_alloc/free/to_fd` trong `libfastrpc_pmem.so` | `LD_PRELOAD=/mnt/lib64/libfastrpc_pmem.so` |
| `HTP Config option: 6 not supported` (err 1000) | Genie luôn bật `IO_MEM_ESTIMATION` (dành cho nhiều PD) | shim `genie_htp_shim.c` (cài bằng `genie_htp_shim.sh`) bỏ đúng option đó |
| ViT `parameter height 32 and width 32 does not match` | Genie 1.18 kiểm tra kiểu input đã cắt patch (Qwen), ViT InternVL nhận NCHW | bỏ `vision-param` khỏi config ViT |
| LLM part 4 err 1004 khi ViT cùng NSP | engine thứ hai bị đặt sang PD mới, PD thứ 2 không chạy được trên board | ViT trên NSP1 (`device_id: 1`, `CDSP1_LIBRARY_PATH`), LLM trên NSP0 |

Crash trên QNX ghi core vào `/dev/shmem` (RAM): luôn `ulimit -c 0`. Demo từ PC: `python board/vlm_genie_demo.py --video
<mp4> --times 0 5 9.3` (nạp model một lần cho mọi frame, kết quả + video trong `outputs/vlm_board_demo/`).

| InternVL3.5-2B W4A16, ảnh 448×448 (256 token ảnh), prompt 281 token | Board SA8650P |
|---|---|
| Nạp model (một lần) | 9.2 s (ViT 1.7 s, LLM 7.8 s) |
| Ảnh vào + ViT (NSP1) | 0.59 s |
| Đọc prompt / token đầu tiên (NSP0) | 1,870 token/s / **0.15 s** |
| Sinh token | 29.5 token/s |
| Câu trả lời 24–56 token, tổng | 0.9–2.0 s (TB 1.54 s) |

Token đầu tiên (action) sau ~0.74 s kể từ khi có ảnh: trong ngân sách 1 s. Model gốc chưa fine-tune: trên clip va chạm AU
nó nhận ra xe máy ngã nhưng vẫn khuyên KEEP khi xe bạc cắt làn ở 24.8 s, cần fine-tune như v3.

**VLA trên board với model mặc định (13/10/2026, `board/vla_genie_demo.py`):** perception + safety gate trên PC,
InternVL3.5-2B trên board nhận ảnh + đúng prompt quyết định của project và trả JSON action; LLM bên trong InternVL
(Qwen3-1.7B, `genie_config.json` chỉ chữ, dùng chung `part*.bin`) viết lời giải thích tiếng Việt từ sự kiện sau gate.
Giải mã greedy (`--setup` ghi `greedy-*.json`). Clip va chạm AU, 6 frame: action có sau ViT 0.59 s + token đầu 0.31 s
(prompt ~600 token), JSON đầy đủ 1.8–2.0 s; lời giải thích 2.9–4.7 s (30 token/s). Model gốc **luôn trả KEEP / risk low**
(kể cả lúc xe cắt làn), đôi khi chép nguyên dòng menu (`"KEEP = hold current speed"`); gate vẫn ra FCW/AEB đúng lúc.
Lời giải thích đọc hiểu được nhưng lẫn chữ Hán và có chi tiết bịa: cần fine-tune (scripts/pipeline_internvl.sh).

**Qwen2.5-VL-7B (bundle AI Hub) không chạy được trên board:** LLM 5 phần × ~826 MB trên DSP, một PD (không gian địa chỉ 32-bit)
chỉ chứa được 3 phần (~2.5 GB); backend mở PD thứ 2 nhưng nạp lỗi 1004. Model VLM trên board này phải ≤ ~2.5 GB trên DSP.

## Tối ưu VLM trên board (13–14/10/2026)

Đo bằng `scripts/bench_board_vlm.py`: cùng prompt quyết định của project (679 token: 108 + 256 token ảnh + 315), 8 frame
`highway_traffic.mp4` × 2 lượt, trung vị. **Mỗi quyết định: 2.50 s → 1.89 s (−24 %)**, cùng action trên mọi frame.

| Bước | Thời gian / quyết định (PC) | Trên board |
|---|---|---|
| Ban đầu (JSON đủ + reason, poll, ảnh float32) | 2.50 s | ViT 576 ms, token đầu 892 ms, tổng 2269 ms, 39 token |
| Context 1024 thay 4096 | 2.48 s | không đổi: Genie đã tự dùng graph cl nhỏ nhất đủ chỗ |
| `poll: false` | 2.40 s | −3.5 % và CPU không còn busy-wait |
| Dừng trước `"reason"` (`stop-sequence`, config `fast`) | 2.10 s | tổng 1867 ms, 22 token, parse 15/15 |
| Ảnh uint8 (giao thức VLM2) + tất cả các bước trên | **1.89 s** | tổng 1813 ms; ngoài server 0.23 → 0.08 s |

Còn lại: ViT 0.57 s (30 %), đọc prompt 0.31 s, sinh 22 token 0.93 s, PC + mạng 0.08 s.
`fast-text-generator-nsp1.json` là mặc định (`scripts/run_demo.sh board`); cần reason trên HUD: `board-reason`.

**Bundle AI Hub đã có sẵn** (đọc từ `qnn-context-binary-utility` và code recipe `qai_hub_models`): attention một head (SHA),
Linear → Conv 1×1, eager attention, KV 8 bit với K chuyển vị và graph chỉ trả KV mới, embedding tách thành LUT, LM head
8 bit, trọng số W4, O3 + VTCM 8 MB, weight sharing giữa graph AR128/AR1 × cl512/1024/2048/4096. Không gian residual của LLM
được **quay bằng ma trận Hadamard** 2048×2048 (kiểu QuaRot/SpinQuant, dàn outlier trước khi lượng tử W4):
`embedding_weights.raw = embed_tokens · R` (sai số 2.6e-8) và ViT xuất `features · R`.

**ViT tự build** (`scripts/export_internvl_vit.py`: graph như recipe AI Hub, R gộp vào Linear cuối của projector,
63 frame calib lái xe; QAIRT 2.46 W8A16 per-channel, O3 + VTCM 8), 17 frame đánh giá, so với float:

| ViT | Board (một mình, NSP0) | SQNR vs float | Cosine token TB |
|---|---|---|---|
| AI Hub `vision_encoder.bin` | **569 ms** | 23.3 dB | 0.9964 |
| Tự build W8A16 | 588 ms | 23.1 dB | 0.9963 |
| + DLBC (`dlbc: 1`) / + `dlbc_weights` | 594 ms | đầu ra trùng từng bit | |
| Tự build W8A8 | **98.7 ms** | −0.4 dB (hỏng) | 0.33 |

Thay ViT tự build vào Genie (`IMG=img-enc-own.json`) chạy được, cùng quyết định; mặc định vẫn là ViT AI Hub (nhanh hơn 3 %).
DLBC không có lợi trên SA8650P: cũng thử trên YOLOP lane (4.19 → 4.17 ms, trong nhiễu, đầu ra trùng từng bit).
W8A8 nhanh gấp 5.8 lần nhưng activation 8 bit phá hỏng ViT (outlier): hướng tiếp theo là mixed precision (A8 cho phần
lớn, 16 bit cho tensor nhạy cảm, phân tích SQNR từng lớp như với YOLO11s).

## adas-vlm-v3 (Qwen2.5-VL-3B fine-tune) tự build bằng QAIRT (23/10/2026)

Model quyết định tốt nhất của project, trước đây chỉ chạy trên GPU của PC, nay được build từ đầu bằng QAIRT 2.46 (không qua
AI Hub: AI Hub không có Qwen2.5-VL-3B, và recipe của nó cần GPU khoảng 40 GB). Quy trình và lệnh: `scripts/v3_qnn.py`,
`scripts/v3_qnn_build.sh`, chi tiết trong `CLAUDE.md`. Binary build cho **SA8650P (HTP v73, soc 52)** và **SA8797P (HTP v81,
soc 72)**, chạy và đo trên SA8650P (chưa có board SA8797P).

| Thành phần | Dạng trên HTP | SA8650P / SA8797P |
|---|---|---|
| ViT 32 block + merger, ảnh 560×308 → 880 patch → 220 token | W8A16 theo kênh, residual xoay trực giao 1280 | 658 / 660 MiB |
| LLM 36 layer chia 4 phần, mỗi phần 2 graph AR128 + AR1 (cl 1024) dùng chung trọng số | **GPTQ W4** theo kênh + A16, KV 8 bit, residual xoay Hadamard | 3 × 344 + 647 MiB (LM head 8 bit ở phần 4) |
| Embedding | LUT float32 trên CPU (`embed_tokens · R`) | 1.24 GB RAM |

Ba chỗ phải sửa so với cách làm của bundle InternVL, mỗi chỗ được đo trước khi chọn:

| Vấn đề | Đo được | Cách sửa |
|---|---|---|
| Mask attention −100 không đủ | logits lệch 0.87 so với transformers (layer 27: logit token "tương lai" cao hơn max hợp lệ +391) | mask −1000 (encoding [−1000, 0]) |
| W4 làm tròn (RTN) phá quyết định | trùng float 48 % (LPBQ-64 RTN 56 %), KV 8 bit một mình 98 % | GPTQ W4 theo kênh: trùng float 98 % |
| Residual ViT có outlier 2.8·10⁴ | ViT W8A16 SQNR 12.8 dB | xoay residual ViT: 20.4 dB (ViT bf16 của PC chỉ 15.2 dB so với fp32) |

Kỳ vọng độ chính xác (mô phỏng trên PC, 900 mẫu val, chưa tính activation 16 bit và ViT 8 bit):

| 900 mẫu val | joint | under-braking | cùng quyết định với float |
|---|---|---|---|
| v3 float (bf16, dạng triển khai, greedy như Genie) | 77.3 % | 10.8 % | – |
| v3 W4 GPTQ + KV 8 bit (cấu hình board) | 76.1 % | 9.8 % | 96.1 % |
| v3 nf4 trên PC (`adas-vla eval`, cách chạy cũ) | 72.6–73.1 % | 8.8–9.1 % | 87–89 % |
| v3 float nhưng ViT float32 thay vì bf16 | 76.9 % | 10.8 % | 98.8 % (ViT bf16 chỉ 15 dB so với float32 mà vẫn ít ảnh hưởng) |

Chạy trên board (Genie, ViT NSP0 + LLM NSP1, server `vlm_genie_server --qwen 308x560`, giao thức VLM3):

```bash
MODEL=v3 bash board/vlm_server.sh deploy     # copy data/qnn_v3/bundle/sa8650 vào /data/adas_vla_genie/adas_vlm_v3
MODEL=v3 bash board/vlm_server.sh start
.venv/bin/python scripts/v3_qnn.py eval-board # 900 mẫu val, metric như adas-vla eval + thời gian từng mẫu
MODEL=v3 bash board/vlm_server.sh stop
```

## Giao thức

Xem đầu file `vla_stream_server.cpp` ("VLASTREAM 1"). `third_party/stb_image.h` = stb_image v2.30 (public domain /
MIT, nothings/stb), sha256 `594c2fe3…00b3`.
