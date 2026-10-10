# Ngân sách thời gian (B0) và đo latency từng stage (B6/B7)

> Bảng ngân sách dưới đây là **đề xuất ban đầu** cho project học tập; chủ repo chốt. Deadline D là mục tiêu
> trên SoC (SA8797P / SA8650P); trên PC, bảng tổng kết dùng chúng để thấy biên dư và độ giật (jitter), không phải
> kết luận đạt / không đạt trên target. Số trong config: mục `budget:` của `configs/default.yaml`.

## 1. Bảng ngân sách

| Tác vụ | Key trong log | Loại realtime | Chu kỳ T | Deadline D | Nơi chạy trên SoC | Chỉ tiêu accuracy |
|---|---|---|---|---|---|---|
| Thu ảnh + letterbox | `read` | firm | 33 ms | 5 ms | ISP / CPU, zero-copy | – |
| Detector YOLO11s + ByteTrack | `detector` | firm | 33 ms | 15 ms | HTP (INT8 / W8A16) + CPU | mAP ≥ FP32 − 1 điểm; recall xe gần ≥ FP32 |
| Lane YOLOP + fit | `lanes` | firm | 33–66 ms | 10 ms | HTP + CPU (hoặc T = 2 frame) | sai số offset_norm ≤ 0.1 |
| Khoảng cách / TTC / cut-in | `geometry` | firm | 33 ms | 3 ms | CPU, C++ | TTC theo golden perception |
| **Safety gate** | `gate` | **hard** | 10 ms | 1 ms | CPU cô lập / safety island | 206/206 golden vectors |
| **Controller** | `control` | **hard** | 10 ms | 0.5 ms | cùng core với gate | golden vectors |
| Cả đường firm (capture → lệnh điều khiển, không tính VLM) | `frame` | firm | 33 ms | 33.3 ms | – | DMR ≤ 1 %, không 2 miss liên tiếp |
| VLM Qwen2.5-VL-3B, 220 token ảnh | log VLM | soft | sự kiện / 15 frame | p50 ≤ 1 s, tuổi ≤ 2 s | HTP (ViT + LM) hoặc GPU | under-braking ≤ 1 %, joint ≥ 85 % |
| LLM giải thích | – | best-effort | theo yêu cầu | – | HTP (Genie) | – |

## 2. Đo như thế nào

`adas-vla run --log run.jsonl` ghi vào mỗi dòng (mỗi frame) một dict `timing` (ms), `vlm_age_s` và
`vlm_age_wall_s`. Cuối run in bảng tổng kết; `adas-vla latency --log run.jsonl [--json stats.json]` tính lại
offline (không cần GPU). Định nghĩa các key nằm trong `adas_vla/timing.py`:

| Key | Ý nghĩa |
|---|---|
| `read` | đọc / decode frame (chỉ với file video; với camera live phần lớn là thời gian chờ frame nên không ghi) |
| `detector` = `det_pre` + `det_model` + `det_post` + `det_track` | phân chia của Ultralytics; `det_track` = ByteTrack + chuyển box |
| `lanes` = `lane_pre` + `lane_model` + `lane_post` | letterbox + normalize / YOLOP forward (có sync GPU) / argmax + resize mask + Hough + fit |
| `geometry`, `gate`, `control` | MotionEstimator, SafetySupervisor.arbitrate, Controller |
| `vlm` | thời gian VLM giữ vòng lặp frame: cả lời gọi ở mode `sync`, chỉ copy frame ở `async` / `process` |
| `frame` | capture → lệnh điều khiển, **không** tính VLM (đường firm, so với chu kỳ frame) |
| `e2e` | capture → lệnh điều khiển, tính cả lời gọi VLM blocking (mode `sync`) |
| `vlm_age_s` | tuổi quyết định VLM đang dùng, theo timeline của video (gate dùng số này) |
| `vlm_age_wall_s` | cùng đại lượng, theo đồng hồ thật từ lúc capture frame mà VLM đã nhìn |

Quy ước thống kê: bỏ `budget.warmup_frames` (30) frame đầu; percentile nearest-rank (p99 là giá trị thật đã xảy
ra); DMR = tỉ lệ frame vượt D; `consec` = chuỗi miss liên tiếp dài nhất.

**Lưu ý timeline vs wall clock.** Với file video, timeline là `idx / fps`. Nếu pipeline chạy chậm hơn real-time,
tuổi quyết định theo timeline **nhỏ hơn** tuổi thật trên camera live (ví dụ dưới: gate thấy p50 2.08 s, thật là
2.62 s). Số dùng để đánh giá realtime là `vlm_age_wall_s`.

## 3. Số nền trên PC (2026-10-10)

RTX 4060 Laptop 8 GB, i7-13650HX; `data/samples/highway_traffic.mp4` (1280×720, 30 fps), 600 frame, bỏ 30 frame
đầu. VLM = `models/adas-vlm-v3` 4-bit. Đơn vị ms, trừ dòng VLM.

| Stage (p50 / p99) | Không VLM | VLM `async` (thread) | VLM `process` |
|---|---|---|---|
| detector | 6.2 / 11.8 | 8.8 / **201** | 19.1 / 38.8 |
| lanes | 15.4 / 17.5 | 18.3 / 47.9 | 30.0 / 42.9 |
| geometry + gate + control | < 0.1 | < 0.1 | < 0.1 |
| frame (D = 33.3) | 22.3 / 28.7, DMR 0.4 % | 28.1 / 222, DMR 14 % | 49.8 / 81.9, DMR 95.6 % |
| FPS | 31.0 | 19.9 | 15.6 |
| VLM latency p50 / p95 (s) | – | 1.46 / 2.92 | 0.96 / 1.03 |
| Tuổi VLM wall p50 (s); % frame quá 2 s hoặc chưa có | – | 2.62; 90.5 % | 1.95; 47.4 % |

Gate và controller (Python) dưới 0.05 ms p99, dư rất xa D.

## 4. Phát hiện

1. **Lane là stage vượt ngân sách lớn nhất.** Ban đầu `lanes` p50 23.3 ms (frame DMR 31 %): normalize float32
   trên CPU mất ~5 ms. Đã chuyển normalize lên GPU (output giống hệt bit-by-bit trên 521 frame của 2 video):
   `lanes` 15.4 ms, frame DMR 0.4 %. Còn lại `lane_model` 7.1 ms + `lane_post` 7.7 ms, trong đó Hough trên mask
   YOLOP 1280×720 (mask dày, ~140 đoạn thẳng) ~7 ms CPU. Trên CPU của SoC phần này sẽ không nhanh hơn ⇒ cần thay
   Hough bằng fit theo hàng trên mask (O(số hàng)) hoặc chạy lane mỗi 2 frame (T = 66 ms, như bảng 1 cho phép).
2. **VLM dùng chung GPU với perception: hai kiểu nhiễu, đúng hai mặt của rủi ro "accelerator không preempt".**
   - Thread (`async`): mỗi lần prefill (~350 ms) làm detector đứng ~240 ms. Microbenchmark (detector ở main
     thread, thread phụ chạy một phần của lời gọi VLM):

     | Thread phụ | detector p50 | p99 | frame > 50 ms |
     |---|---|---|---|
     | không có | 6.6 | 13.4 | 0 |
     | chỉ pre-process VLM (CPU) | 8.9 | 19.2 | 0 |
     | chỉ prefill (GPU), default stream | 19.9 | 246.7 | 40 |
     | chỉ prefill, CUDA stream riêng | 18.4 | 241.4 | 49 |
     | **prefill ở process riêng** | 21.3 | **38.2** | **0** |

     CUDA stream riêng (kể cả perception ở stream ưu tiên cao) không đỡ; process riêng thì hết spike ⇒ thread
     VLM giữ GIL trong lúc chờ GPU. Trong run thật, 16/17 spike > 150 ms rơi 2–4 frame sau khi có kết quả VLM
     mới (spike còn lại nằm trước kết quả đầu tiên), tức lúc lời gọi kế tiếp bắt đầu prefill.
   - Process (`process`, mới): hết spike, VLM nhanh hơn và tươi hơn, nhưng GPU chia time-slice giữa hai CUDA
     context nên perception chậm đều ~2 lần.
   - Hệ quả cho SoC: không được để VLM và CNN perception chung một hàng đợi accelerator mà không có cơ chế ưu tiên.
     Hướng: VLM trên accelerator / HTP session riêng có priority thấp hơn, hoặc chia prefill thành từng đoạn chạy
     trong khoảng trống giữa các frame perception; đo lại bằng đúng bảng này trên board.
3. **VLM gọi liên tục vẫn không giữ được tuổi ≤ 2 s.** Với latency L và các lời gọi nối tiếp, tuổi dao động
   từ L đến ~2L, nên muốn tuổi ≤ 2 s thì cần L ≤ ~1 s *khi chạy chung tải*; trên PC chỉ đạt ở mode `process`
   (p50 0.96 s), và khi đó perception vỡ ngân sách.

## 5. Việc tiếp theo

- Lane: fit theo hàng thay cho Hough (so sánh offset_norm với bản hiện tại trên cùng clip) hoặc T = 2 frame.
- `run --realtime` cho file video: bỏ frame để timeline bám đồng hồ thật như camera live, để gate dùng đúng tuổi.
- Port C++ gate + controller (golden vectors sẵn sàng), đo WCET bằng vòng lặp 10⁵ lần.
- Trên board: cùng bảng này với `det_*` / `lane_*` thay bằng thời gian QNN (`qnn-net-run` / profiling HTP).
