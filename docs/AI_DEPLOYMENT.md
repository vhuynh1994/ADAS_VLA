# Triển khai mô hình AI trên hệ thống nhúng ADAS: quy trình, tối ưu, realtime

> Tài liệu học tập / nghiên cứu cho hai repo **ADAS_VLA** (pipeline camera → perception CNN → VLM → safety gate →
> controller, đích Snapdragon Ride Elite SA8797P) và **ADAS_CNN** (GTSRB / KITTI → ONNX → INT8 → Qualcomm, lộ trình
> M0–M8). Phần realtime tập trung vào **CNN / RNN / DNN**; LLM / VLM chỉ được nhắc ở mức "tác vụ soft realtime"
> và sẽ có tài liệu riêng.
>
> Quy ước: **công thức** viết trong khối `code`; **tiêu chí hoàn thành (DoD)** là điều kiện kiểm chứng được bằng số
> đo; mọi con số mục tiêu trong tài liệu là **đề xuất ban đầu** cho project học tập, không phải số đã đo trên board
> (xem mục 6 để biết số nào đã đo).

Mục lục

1. [Khung tư duy: 4 tài nguyên, 3 câu hỏi](#1-khung-tư-duy-4-tài-nguyên-3-câu-hỏi)
2. [Quy trình triển khai từng bước (B0 → B9)](#2-quy-trình-triển-khai-từng-bước)
3. [Phương pháp tối ưu và phân loại](#3-phương-pháp-tối-ưu-và-phân-loại)
4. [Realtime với mô hình CNN / RNN / DNN](#4-realtime-với-mô-hình-cnn--rnn--dnn)
5. [Áp dụng: ADAS_VLA và ADAS_CNN](#5-áp-dụng-adas_vla-và-adas_cnn)
6. [Trạng thái số liệu và việc cần đo tiếp](#6-trạng-thái-số-liệu-và-việc-cần-đo-tiếp)
7. [Checklist tổng hợp](#7-checklist-tổng-hợp)
8. [Tài liệu tham khảo (công khai)](#8-tài-liệu-tham-khảo-công-khai)

---

## 1. Khung tư duy: 4 tài nguyên, 3 câu hỏi

Mọi bước deploy đều xoay quanh **bốn tài nguyên hữu hạn** của thiết bị đích và **ba câu hỏi** phải trả lời bằng số:

| Tài nguyên | Đơn vị đo | Trên SA8797P / board Qualcomm cần biết |
|---|---|---|
| Tính toán (compute) | MAC/s, TOPS (INT8), FLOPS (FP16) | peak TOPS của HTP, số core HTP, clock profile |
| Băng thông bộ nhớ | GB/s (DDR), GB/s (VTCM/L2 nội bộ NPU) | băng thông DDR dùng chung với camera/ISP/GPU |
| Dung lượng bộ nhớ | MB (weights + activations + arena), MB/process | giới hạn bộ nhớ mỗi process trên HTP |
| Năng lượng / nhiệt | W, J/inference, °C, clock sau throttle | TDP, ngưỡng throttle, chế độ perf lock |

Ba câu hỏi, mỗi câu phải có số đo kèm điều kiện đo:

1. **Đúng không?** (functional): accuracy / mAP / under-braking so với mô hình tham chiếu FP32, trên **cùng** tập val.
2. **Kịp không?** (temporal): latency p50/p99, WCET quan sát được, deadline-miss ratio, jitter, tuổi dữ liệu (data age).
3. **Vừa không?** (resource): peak memory, model size, băng thông, J/inference, nhiệt sau ≥ 30 phút chạy liên tục.

Nguyên tắc xuyên suốt của cả hai repo: **đo, không đoán** (ADAS_CNN "Không số giả"); **safety gate deterministic
quyết định, mạng neural chỉ khuyến nghị** (ADAS_VLA). Hai nguyên tắc này quyết định cách phân loại realtime ở mục 4.

---

## 2. Quy trình triển khai từng bước

Mỗi bước có: mục tiêu, đặc điểm/đặc trưng, đầu vào → đầu ra, **công thức đo**, **DoD**. Bước nào không đạt DoD thì
quay lại bước trước, không "đi tiếp rồi sửa sau" (lỗi lượng tử hóa phát hiện trên board đắt gấp nhiều lần phát hiện
trên host).

```
B0 yêu cầu ─► B1 baseline FP32 ─► B2 export/đóng băng graph ─► B3 nén ─► B4 lượng tử hóa
      ─► B5 biên dịch cho target ─► B6 tích hợp runtime ─► B7 đo trên target ─► B8 kiểm chứng hệ thống ─► B9 vận hành
```

### B0. Chốt yêu cầu hệ thống (requirements & budget)

**Mục tiêu.** Biến "chạy realtime trên xe" thành các con số có thể kiểm chứng.

**Đặc trưng.** Là bước duy nhất không cần code; nhưng mọi DoD về sau tham chiếu về đây. Gồm: miền vận hành (ODD:
tốc độ, ngày/đêm, mưa), chu kỳ cảm biến, deadline từng tầng, ngân sách bộ nhớ/năng lượng, mức an toàn (QM / ASIL)
cho từng thành phần, chỉ tiêu accuracy an toàn (với ADAS_VLA: **under-braking** trước, accuracy sau).

**Công thức.**

```
Chu kỳ khung hình:      T = 1 / f_camera            (30 fps → T = 33.3 ms)
Ngân sách chuỗi nối tiếp: Σ_i C_i ≤ D_e2e           (C_i = thời gian xấu nhất của tầng i, D_e2e = deadline sensor→actuator)
Khoảng dừng (stopping distance) để suy ra deadline phản ứng:
    d_stop = v·t_react + v² / (2·a_brake)          (v m/s, a_brake ≈ 6–8 m/s² mặt đường khô)
    → phần t_react do hệ thống (sensor → lệnh phanh) phải nhỏ hơn nhiều so với t_react con người (~1–1.5 s);
      ngân sách thường đặt ≤ 100–150 ms cho AEB.
Headroom:               H = (D − WCET_obs) / D       (mục tiêu H ≥ 0.2–0.3 ở B7)
```

**DoD B0.** Có bảng "ngân sách" cho từng tầng với 5 cột: chu kỳ T, deadline D, loại realtime (hard/firm/soft, mục 4),
tài nguyên được phép (core/accelerator, MB), chỉ tiêu accuracy. Với ADAS_VLA bảng này ở mục 5.1.

### B1. Mô hình tham chiếu FP32 (golden model)

**Mục tiêu.** Có một mô hình, một tập val cố định và một bộ metric làm **chuẩn so sánh** cho mọi biến thể tối ưu.

**Đặc trưng.** Tập val phải chia theo *nhóm* (route / drive / track / clip) để không rò rỉ (ADAS_VLA:
`data/ds_v2/labels.splits.json`; ADAS_CNN: GTSRB theo track, KITTI theo drive). Lưu luôn **đầu ra thô** (logits /
heatmap / box trước NMS) của một subset nhỏ (50–200 mẫu) làm *golden outputs* cho B2, B4, B5.

**Công thức.**

```
Top-1 accuracy      = đúng / tổng
Precision, Recall   = TP/(TP+FP), TP/(TP+FN);  F1 = 2PR/(P+R)
AP (một lớp)        = ∫₀¹ p(r) dr  (nội suy theo protocol COCO 101 điểm hoặc KITTI 40 điểm);  mAP = trung bình các lớp
Under-braking rate  = #{nhãn ∈ {DECELERATE, BRAKE, STOP} và dự đoán phanh yếu hơn nhãn} / n     (ADAS_VLA)
Khoảng tin cậy Wilson 95% cho một tỉ lệ p̂ = k/n:
    center = (p̂ + z²/2n) / (1 + z²/n),   half = z·sqrt(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n),   z = 1.96
    (đã cài trong adas_vla/training/evaluate.py: wilson_interval)
```

Vì sao cần khoảng tin cậy: với n = 900 mẫu, một chênh lệch 1.5 điểm % có thể nằm trong nhiễu; hai biến thể chỉ
được coi là khác nhau khi khoảng tin cậy không chồng lấn (hoặc test McNemar trên cùng mẫu).

**DoD B1.** (1) Tập val đóng băng (hash file split); (2) metric + CI được ghi vào JSON có kèm git commit, phiên bản
thư viện, GPU; (3) golden outputs lưu kèm hash; (4) với ADAS_VLA: báo under-braking trước accuracy.

### B2. Đóng băng graph và export (ONNX / TorchScript)

**Mục tiêu.** Biến mô hình huấn luyện (Python, shape động, op tuỳ ý) thành graph **tĩnh**, chỉ dùng op mà target hỗ trợ.

**Đặc trưng.**
- Shape tĩnh (`[1,3,384,640]` cho detector; `[2,3,308,560]` cho ViT 2 frame của VLM). Shape động là nguồn
  không xác định lớn nhất về thời gian (mục 4).
- Folding BN vào conv; loại bỏ các op "Python" (NMS, decode, letterbox) ra khỏi graph → làm trên CPU, đo riêng.
- Kiểm tra **op coverage**: histogram op trong graph đối chiếu bảng op được backend hỗ trợ (QNN HTP, SNPE, TensorRT).
- Kiểm tra **parity** (tương đương số) giữa framework và runtime ONNX.

**Công thức.**

```
Sai số tuyệt đối lớn nhất:  e_max = max |y_ref − y_onnx|
Sai số tương đối:          e_rel = ‖y_ref − y_onnx‖₂ / ‖y_ref‖₂
Cosine similarity:         cos = (y_ref · y_onnx) / (‖y_ref‖·‖y_onnx‖)
Tỉ lệ đồng thuận quyết định: agree = #{argmax_ref = argmax_onnx} / n   (classification) ; |ΔmAP| (detection)
MACs của một conv:         MACs = H_out · W_out · C_out · (K_h · K_w · C_in / groups);   FLOPs ≈ 2·MACs
```

**DoD B2.** `e_max < 1e-3` (FP32), `cos > 0.9999`, `|Δaccuracy| ≤ 0.1 điểm %` trên val (ADAS_CNN M5
`parity_ok=True`); 0 op nằm ngoài bảng hỗ trợ của backend đích **trên đường nóng** (hot path); input/output đã ghi rõ
layout (NCHW/NHWC), dải giá trị (0..1 hay 0..255), mean/std; MACs và số tham số được in ra và lưu.

### B3. Nén mô hình (compression) – tuỳ chọn, trước lượng tử hóa

**Mục tiêu.** Giảm compute / tham số **khi và chỉ khi** B7 cho thấy không đủ ngân sách; không nén "cho đẹp".

**Đặc trưng.** Có 4 họ (chi tiết mục 3.3): pruning (có cấu trúc / không cấu trúc), distillation, phân rã hạng thấp,
tìm kiến trúc (NAS / thay backbone, giảm độ phân giải, width multiplier). Pruning không cấu trúc **không** nhanh hơn
trên NPU thông thường (không có hỗ trợ sparse); pruning theo kênh mới giảm latency thật.

**Công thức.**

```
Sparsity           s = #weights = 0 / #weights
Tỉ lệ nén          CR = size_gốc / size_sau
Giảm compute       ΔMACs = 1 − MACs_sau / MACs_gốc        (chỉ là dự đoán; latency phải đo ở B7)
Distillation loss  L = (1−α)·CE(y, p_student) + α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))
Giảm độ phân giải  MACs ∝ H·W  → giảm cạnh 384×1280 → 256×832 là ≈ ×0.43 compute, nhưng recall vật nhỏ giảm
```

**DoD B3.** Mỗi biến thể nén có **một dòng** trong bảng kết quả với: accuracy (+CI), MACs, tham số, latency đo thật
trên host **và** (sau B7) trên target; kết luận "nhanh hơn" chỉ được viết khi latency đo giảm.

### B4. Lượng tử hóa (quantization)

**Mục tiêu.** Chạy ở INT8 (W8A8 / W8A16) trên HTP; VLM ở W4A16 (sau). Giữ accuracy trong ngân sách B0.

**Đặc trưng.**
- **PTQ** (post-training): nhanh, cần tập calibration 100–1000 mẫu lấy từ **train** (không bao giờ từ test), đại
  diện cho ODD (đêm, mưa, đô thị, cao tốc).
- **QAT** (quantization-aware training): khi PTQ rớt quá ngân sách; dùng fake-quant + straight-through estimator.
- Lựa chọn: đối xứng / bất đối xứng, per-tensor / per-channel (weights), phương pháp calibration (min-max, percentile,
  entropy/KL, MSE), lớp nào giữ FP16 (lớp đầu/cuối, softmax, lớp nhạy).
- Hexagon HTP ưa INT8 đối xứng cho weights, hoạt động tốt với **W8A16** khi activation có dải rộng.

**Công thức.**

```
Lượng tử hoá affine:  q = clamp( round(x / s) + z , q_min, q_max )      x̂ = s · (q − z)
   bất đối xứng:      s = (x_max − x_min) / (q_max − q_min),  z = round(q_min − x_min / s)
   đối xứng (INT8):   s = max|x| / 127,  z = 0
Sai số lượng tử hoá:  ε = x − x̂,  MSE = mean(ε²)
SQNR (dB)           = 10·log10( Σ x² / Σ (x − x̂)² )          (mỗi tensor; < ~20 dB là lớp "có vấn đề")
Calibration entropy: chọn ngưỡng T* = argmin_T KL( P_fp32 ‖ Q_int8(T) )
Ngân sách rớt accuracy: ΔAcc = Acc_fp32 − Acc_int8 ≤ budget  (đề xuất: ≤ 0.5 điểm % classification;
                       ≤ 1 điểm mAP detection; với ADAS_VLA thêm: under-braking không tăng quá CI của FP32)
```

**DoD B4.** Bảng FP32 vs INT8-PTQ (≥ 2 phương pháp calibration) vs INT8-QAT (nếu cần) trên **cùng val**, mỗi ô
có CI; danh sách lớp bị giữ FP16 và lý do (SQNR / sensitivity); file encodings (scale/zero-point) được lưu cùng
ONNX; calibration **không** chứa mẫu test (script ADAS_CNN từ chối split test).

### B5. Biên dịch / chuyển đổi cho target (QNN / SNPE / TensorRT / TFLite)

**Mục tiêu.** Sinh artefact chạy được trên accelerator đích (QNN context binary, SNPE DLC, TensorRT engine) với
**0 op rơi về CPU** trên đường nóng.

**Đặc trưng.** Converter làm: ánh xạ op → kernel backend, chọn layout nội bộ (HTP dùng layout tiled riêng), phân vùng
graph (phần không hỗ trợ rơi về CPU – mỗi lần rơi là một lần copy + đồng bộ), lập kế hoạch bộ nhớ tĩnh. Artefact gắn
với **phiên bản SDK + kiến trúc HTP (v68…v81)**: đổi SDK là phải đo lại.

**Công thức.**

```
Số phân vùng graph  n_part   (mục tiêu 1);  số op fallback CPU  n_fb  (mục tiêu 0 trên hot path)
Overhead chuyển vùng ≈ Σ (copy_bytes / BW + t_sync)   → nhìn trong profiler per-layer
Kích thước artefact  size_bin  so với ngân sách flash; thời gian load/khởi tạo t_init (ảnh hưởng boot-time)
```

**DoD B5.** Artefact build được từ script trong repo (`scripts/deploy/qualcomm/*.sh` ADAS_CNN, `adas-vla export` +
qairt-converter/quantizer ADAS_VLA) với log converter lưu kèm; `n_fb = 0` hoặc danh sách op fallback có lý giải;
chạy artefact trên **cùng subset golden** của B1: accuracy chênh ≤ 1 điểm % so với ORT INT8 trên host (ADAS_CNN M7).

### B6. Tích hợp vào runtime / pipeline

**Mục tiêu.** Mô hình trở thành **một tác vụ** trong pipeline: nhận frame từ camera, trả kết quả cho tầng sau với
giao diện, bộ nhớ và luồng được xác định.

**Đặc trưng.** Tiền xử lý (letterbox, chuyển màu, chuẩn hóa) và hậu xử lý (decode, NMS, tracker) thường chiếm
**bằng hoặc hơn** thời gian inference INT8; phải đo tách rời (ADAS_CNN `benchmark_pipeline` tách pre / model / post).
Tránh copy: buffer camera (dmabuf) → NPU trực tiếp; tránh cấp phát động trong vòng lặp; tách luồng theo tốc độ
(System 1 mỗi frame, System 2 mỗi N frame như ADAS_VLA).

**Công thức.**

```
Latency end-to-end một frame:  L = t_pre + t_model + t_post + t_copy + t_sync
Throughput (pipeline k tầng song song):  X = 1 / max_i C_i   (tầng chậm nhất quyết định), latency = Σ C_i
Định luật Little:  số frame đang "trong ống" N = λ · L    (λ = fps, L = latency) → N > 1 nghĩa là có pipelining
Tuổi dữ liệu khi ra quyết định:  age = t_decide − t_capture   (phải ≤ ngân sách B0, không chỉ "latency")
```

**DoD B6.** Pipeline chạy liên tục ≥ 10 phút trên host với video thật không rò bộ nhớ (RSS phẳng), log per-frame
có timestamp capture/decide; profile `t_pre / t_model / t_post` được in và lưu; không có `malloc`/`new` trong vòng lặp
nóng của phần C++ (kiểm bằng sanitizer / đếm allocation).

### B7. Đo hiệu năng trên target (profiling & benchmarking)

**Mục tiêu.** Số thật trên board: latency phân phối, bộ nhớ đỉnh, băng thông, năng lượng, nhiệt – đo theo một
**protocol** cố định để so sánh được giữa các biến thể.

**Đặc trưng (protocol, theo ADAS_CNN `docs/09`).** Warm-up (bỏ ≥ 20 lần đầu: JIT, autotune, clock ramp); đồng bộ
thiết bị trước khi dừng đồng hồ; báo **p50 / p95 / p99 / max**, không chỉ mean; cố định số thread và perf profile;
đo model-only **và** end-to-end; đo ở trạng thái nhiệt ổn định (≥ 30 phút) chứ không chỉ lúc board lạnh.

**Công thức.**

```
Percentile:   p_k = giá trị mà k % mẫu ≤ nó;  jitter J = p99 − p50 (hoặc max − min)
Số mẫu để tin p99:  cần ≥ ~30 mẫu nằm trên p99  → n ≥ 3000 lần chạy (quy tắc ngón tay cái)
Utilization compute:  U_c = (MACs / t_model) / Peak_MAC/s
Roofline:   I = FLOPs / Bytes_truy_cập (arithmetic intensity, FLOP/byte)
            Perf_đạt được ≤ min( Peak_FLOPs ,  BW · I )
            Ridge point I* = Peak_FLOPs / BW ;  I < I*  → memory-bound (tối ưu dữ liệu/bộ nhớ, mục 3.1–3.2)
                                                 I > I*  → compute-bound (tối ưu kernel/precision, mục 3.3)
Năng lượng:  E_inf = P_avg · t_model   (J/inference);  hiệu suất = inferences / J ;  TOPS/W
Bộ nhớ:      peak_RSS, arena NPU, weights + activations (so với ngân sách B0)
Nhiệt:       Δclock = clock_lạnh − clock_sau_30_phút;  latency_sau_30_phút / latency_lạnh (mục tiêu ≤ 1.1)
```

**DoD B7.** JSON kết quả theo schema cố định (`experiments/results/*_htp.json` ADAS_CNN) với: SoC, SDK, HTP arch,
perf profile, số thread, p50/p95/p99/max, peak memory, (nếu có) P_avg và E_inf, nhiệt độ/clock trước–sau; `WCET_obs`
(max quan sát trong ≥ 3000 lần + 30 phút chạy nóng) ≤ D với headroom ≥ 20 %.

### B8. Kiểm chứng hệ thống (SIL / HIL, regression, an toàn)

**Mục tiêu.** Chứng minh **hành vi hệ thống** (không chỉ model) đúng và đúng giờ trong các kịch bản ODD, gồm cả khi
model sai / chậm / vắng.

**Đặc trưng.** Ba lớp kiểm chứng: (1) **đơn vị + golden vectors** cho phần deterministic (ADAS_VLA
`tests/data/safety_golden.json`, 206 kịch bản, dùng làm **equivalence test** cho bản C++); (2) **replay** log/video
thật qua toàn pipeline (SIL), so sánh quyết định cuối với nhãn (ADAS_VLA `scripts/gate_replay.py capture|replay`: detector +
lane chạy một lần trên GPU, hình học + TTC + gate phát lại trên CPU theo từng biến thể config); (3) **fault injection**: model trả kết quả trễ,
JSON lỗi, mất frame, tracker mất ID → gate phải fallback về luật (ADAS_VLA `max_decision_age_s`).

**Công thức.**

```
Tỉ lệ sai lệch golden:  mismatch = #{case: C++ ≠ Python} / 206   → phải = 0 (so sánh số thực với ε = 1e-6)
System under-braking  = under-braking sau gate (adas-vla eval: system_under_braking_rate) ≤ under-braking VLM
   (gate chỉ được phanh NHIỀU hơn VLM: bất đẳng thức này là một test, không phải nhận xét)
Deadline-miss ratio   DMR = #job trễ / #job   (ngưỡng theo loại realtime, mục 4.4)
Thời gian phản ứng    t_react = t_lệnh_phanh − t_hazard_visible   trên clip Nexar (time_of_alert có sẵn)
```

**DoD B8.** 65+ unit test và golden check xanh trong CI; bản C++ của gate + controller tái tạo 100 % golden vectors;
replay video lái bình thường: tỉ lệ thời gian AEB sai và số lần AEB sai mỗi phút được đo và giảm qua từng phiên bản
(đo 29/09 trên PC bằng `gate_replay.py`: 15,8 % / 8,6 lần mỗi phút → 4,0 % / 2,3 lần mỗi phút; ngưỡng đề xuất dài hạn
cho camera đơn: ≤ 1 lần mỗi giờ AEB, ≤ 6 lần mỗi giờ FCW, nhiều khả năng cần radar/depth fusion để đạt); đồng thời tỉ lệ
AEB trong cửa sổ nguy hiểm Nexar không được giảm thêm so với cửa sổ đối chứng (hiện 53 % so với 25 %); fault-injection:
100 % trường hợp VLM vắng/trễ đều cho ra quyết định từ luật trong đúng chu kỳ.

### B9. Vận hành: phát hành, giám sát, cập nhật

**Mục tiêu.** Model trên xe có phiên bản, có giám sát sức khỏe, có đường cập nhật (OTA) và quay lui.

**Đặc trưng.** Watchdog cho từng tác vụ (quá hạn k lần liên tiếp → degraded mode); telemetry tối thiểu (latency
histogram, DMR, nhiệt, số lần fallback); phát hiện **drift** đầu vào (phân phối độ sáng, số object/frame) và đầu ra
(tỉ lệ action); artefact ký số và ghép với phiên bản SDK/firmware.

**Công thức.**

```
Drift đầu vào:  PSI = Σ_bins (p_now − p_ref) · ln(p_now / p_ref)    (PSI > 0.2: đáng xem lại)
Sức khỏe tác vụ:  DMR cửa sổ trượt 1 phút;  số lần watchdog;  thời gian ở degraded mode / giờ
```

**DoD B9.** Mỗi artefact có manifest (hash model, SDK, HTP arch, config, commit); dashboard/log có 4 chỉ số trên;
kịch bản rollback được tập dượt một lần.

### Bảng tóm tắt bước → thước đo → DoD

| Bước | Thước đo chính | Công thức / cách đo | DoD (đề xuất) |
|---|---|---|---|
| B0 | Ngân sách T, D, MB, W | `Σ C_i ≤ D_e2e`, `H = (D−WCET)/D` | bảng ngân sách từng tầng |
| B1 | Accuracy, under-braking + CI | Wilson 95 % | val đóng băng, golden outputs lưu |
| B2 | Parity, op coverage | `e_max`, `cos`, histogram op | `e_max<1e-3`, 0 op không hỗ trợ |
| B3 | MACs, params, latency thật | `ΔMACs`, benchmark | nhanh hơn **đo được** mới giữ |
| B4 | ΔAcc, SQNR | `q = clamp(round(x/s)+z)`, SQNR | ΔAcc ≤ ngân sách, calibration từ train |
| B5 | n_fallback, Δacc board vs host | profiler per-layer | `n_fb = 0`, Δacc ≤ 1 điểm |
| B6 | t_pre/model/post, age | `L = Σ t`, `age = t_decide − t_capture` | 10 phút không rò, 0 alloc hot path |
| B7 | p50/p99/max, roofline, J/inf, nhiệt | mục B7 | WCET_obs ≤ D, headroom ≥ 20 % |
| B8 | golden mismatch, DMR, t_react | mục B8 | 0 mismatch, fallback 100 % |
| B9 | DMR trượt, PSI, watchdog | mục B9 | manifest + rollback |

---

## 3. Phương pháp tối ưu và phân loại

Trước khi chọn kỹ thuật, **xác định nút thắt** bằng roofline (B7): memory-bound → mục 3.1 và 3.2; compute-bound →
3.3; nhiều model tranh tài nguyên → 3.4; latency nằm ngoài model (copy, IPC, scheduling) → 3.5.

### 3.1 Tối ưu theo đường dữ liệu (dataflow)

Mục tiêu: **giảm số byte di chuyển** và **số lần chờ** giữa các khối, vì trên SoC băng thông DDR dùng chung và một
lần copy frame 1280×720×3 (2.8 MB) đã tốn ~0.3 ms ở 10 GB/s hiệu dụng, cộng thêm đồng bộ cache.

| Phân loại nhỏ | Kỹ thuật | Đo bằng | Ghi chú cho ADAS |
|---|---|---|---|
| **Layout dữ liệu** | NCHW ↔ NHWC, layout tiled của HTP; xoá cặp Transpose đầu/cuối | số op `Transpose` trong graph, bytes/frame | HTP chọn layout nội bộ; để converter xử lý, không chèn transpose trong model |
| **Hợp nhất toán tử** (fusion) | Conv+BN+Act, Conv+Add (residual), elementwise chain | số node sau convert, bytes activation ghi ra DDR | mỗi fusion bớt một lần ghi/đọc activation |
| **Zero-copy** | dmabuf / ION / shared memory từ camera → NPU; map thay vì copy | số copy / frame, `t_copy` | ADAS_VLA: frame BGR numpy copy nhiều lần (PIL, letterbox) – chỉ chấp nhận trên PC |
| **Pipelining / double-buffering** | ping-pong buffer: capture frame k+1 trong khi infer frame k | throughput `X`, latency `L`, N theo Little | tăng throughput, **không** giảm latency; age tăng thêm 1 chu kỳ |
| **Hẹp kiểu dữ liệu đầu vào** | nhận UINT8 trực tiếp, chuẩn hóa trong graph (fold mean/std vào conv đầu) | bytes/frame ÷ 4 | bỏ bước float32 trên CPU |
| **Chọn độ phân giải / ROI** | letterbox 384×640 thay 720p; crop ROI vùng đường | MACs ∝ H·W, recall vật nhỏ | detector ADAS_VLA 384×640; VLM 560×308 cố định |
| **Tiling / streaming** | chia feature map để vừa VTCM/L2, tránh tràn ra DDR | cache miss, bytes DDR | converter HTP làm tự động; biết để đọc profiler |
| **Batching** | gom nhiều ảnh | throughput ↑, latency ↑ | **không dùng** cho đường an toàn (batch = 1); dùng cho offline eval |
| **Bỏ qua / tái dụng theo thời gian** | frame skipping, chạy lane mỗi 2 frame, tracker nội suy | fps hiệu dụng, age | ADAS_VLA: VLM chạy theo sự kiện / mỗi N frame (System 2) |

### 3.2 Tối ưu bộ nhớ

Mục tiêu: **vừa** ngân sách (weights + activations + arena), **không cấp phát động** khi chạy, **ít miss cache**.

| Phân loại nhỏ | Kỹ thuật | Đo bằng | Ghi chú |
|---|---|---|---|
| **Giảm weights** | lượng tử hóa (FP32→INT8 = ×¼), pruning kênh, chia sẻ trọng số, nén entropy khi lưu | `size_bin`, CR | VLM: W4 ≈ 2 GB cho 3B |
| **Giảm activations** | độ phân giải nhỏ, INT8 activation, fusion (ít tensor trung gian) | peak activation (profiler) | activations thường lớn hơn weights ở CNN độ phân giải cao |
| **Lập kế hoạch bộ nhớ tĩnh** | phân tích liveness → tô màu buffer dùng lại (arena một lần) | arena size vs Σ tensor | runtime NPU làm; phần C++ của ta cũng phải: pool cố định |
| **In-place / tái dụng buffer** | ReLU, Add in-place; buffer pre/post cố định | số allocation / frame = 0 | kiểm bằng hook `malloc` hoặc sanitizer |
| **Khóa trang & huge page** | `mlockall`, hugepages cho weights/arena | page fault / s = 0 sau khởi động | page fault giữa chừng = jitter hàng ms |
| **Cache-aware** | tiling vừa L2/VTCM; prefetch; tránh false sharing giữa thread | cache miss rate (perf) | HTP có TCM riêng; CPU: L2 per core |
| **Phân tầng lưu trữ / streaming weights** | model lớn: tải từng tầng từ flash, giữ "nóng" phần dùng mỗi frame | t_init, băng thông flash | áp dụng VLM/LLM, không áp dụng CNN nhỏ |
| **Chia mô hình** | ViT và LM thành hai graph (ADAS_VLA) để vừa giới hạn bộ nhớ/process | peak/process | docs/DEPLOY_SA8797P.md mục 6 |
| **Trạng thái (stateful)** | RNN hidden state, tracker, FrameHistory 0.5 s ring buffer | bytes giữ giữa các frame | ring buffer cố định kích thước = gap/T + slack |

### 3.3 Tối ưu thực thi (compute)

| Phân loại nhỏ | Kỹ thuật | Đo bằng | Ghi chú |
|---|---|---|---|
| **Mức kernel** | SIMD (HVX trên Hexagon), ma trận (HMX), Winograd/FFT conv, im2col vs direct, autotuning kernel | `U_c`, GMAC/s đạt được | do runtime vendor; ta chọn op "thân thiện": conv 3×3/1×1, depthwise, ReLU/ReLU6, không op lạ |
| **Mức graph** | constant folding, dead-node elimination, bỏ transpose/reshape thừa, chọn opset | số node, số layout change | `onnxsim` / `simplify=True` khi export |
| **Độ chính xác số** | INT8, W8A16, FP16, mixed precision theo lớp | ΔAcc, t_model | INT8 ≈ 2–4× FP16 trên NPU |
| **Song song** | intra-op (thread trong kernel), inter-op (các nhánh độc lập), đa accelerator (YOLO trên HTP, lane trên HTP khác hoặc GPU) | `U_c`, số core bận | nhiều thread CPU không giúp nếu bottleneck ở DDR |
| **Lập lịch & tần số** | perf profile (burst/sustained), khóa clock, affinity, ưu tiên | p99, jitter, nhiệt | burst nhanh hơn nhưng throttle; đo ở sustained |
| **Thuật toán / mô hình** | pruning kênh, distillation, NAS, early-exit, cascade (model nhỏ lọc trước), giảm độ phân giải | MACs, ΔAcc, t_model | pruning phi cấu trúc không tăng tốc trên HTP |
| **Khai thác thưa** | sparsity có cấu trúc (2:4) nếu hardware hỗ trợ | t_model | kiểm tra hỗ trợ của SDK trước |
| **Hậu xử lý** | NMS vectorized, decode trên NPU nếu được, giới hạn top-k | `t_post` | NMS trên CPU thường > t_model INT8 |

### 3.4 Tối ưu theo workload

Bước 1: **phân loại workload** – bước này quyết định kỹ thuật được dùng.

| Trục phân loại | Giá trị | Hệ quả thiết kế |
|---|---|---|
| Nút thắt (roofline) | compute-bound / memory-bound | 3.3 vs 3.1–3.2 |
| Mẫu kích hoạt | **chu kỳ** (perception mỗi frame) / **sự kiện** (VLM khi scene đổi) / **theo yêu cầu** (LLM giải thích) | periodic → lý thuyết lập lịch mục 4; event → cần giới hạn tần suất (ADAS_VLA `min_interval_frames`) |
| Mục tiêu | latency-critical / throughput | batch = 1, ưu tiên cao vs batch lớn, ưu tiên thấp |
| Shape | tĩnh / động | động → không thể có WCET; cố định hoặc pad |
| Trạng thái | stateless / stateful (RNN, tracker, latch của gate) | stateful cần reset có kiểm soát (ADAS_VLA `reset()`), và state nằm trong ngân sách bộ nhớ |
| Số model đồng thời | 1 / nhiều (YOLO + YOLOP + VLM) | tranh chấp NPU (không preempt) → mục 4.3 blocking |
| Mức an toàn | QM / ASIL-B / ASIL-D | quyết định nơi chạy (3.5) và loại realtime (4) |

Bước 2: kỹ thuật theo workload.

- **Dual-rate (System 1 / System 2)**: tác vụ nhanh, nhỏ, deterministic chạy mỗi frame; tác vụ chậm chạy thưa và
  kết quả có **tuổi tối đa** (`max_decision_age_s`), quá tuổi thì bỏ, dùng luật. Đây là cách ADAS_VLA biến VLM
  (soft) thành an toàn với gate (hard).
- **Admission control / rate limiting**: không cho tác vụ sự kiện nổ tần suất (ví dụ scene đổi liên tục).
- **Phân vùng tài nguyên**: core CPU riêng cho gate+controller (`isolcpus`, affinity); NPU theo thứ tự ưu tiên hàng
  đợi (tác vụ an toàn vào trước); GPU cho HUD/hiển thị.
- **Degraded modes**: mất lane → corridor mặc định; mất VLM → ACC luật; mất detector → cảnh báo & yêu cầu lái.
- **Co-scheduling theo thời gian**: xếp inference YOLO và YOLOP lệch pha để không chờ nhau trên cùng NPU, hoặc gộp
  hai graph thành một lần gọi nếu runtime cho phép.

### 3.5 Tối ưu theo kiến trúc tích hợp

| Phân loại nhỏ | Lựa chọn | Đo bằng | Áp dụng |
|---|---|---|---|
| **Ánh xạ dị thể** (mapping) | model nào trên CPU / GPU / NPU-HTP / DSP / MCU safety island | `U` từng đơn vị, t_e2e | ADAS_VLA: CNN + ViT trên HTP; tracker/geometry/gate/controller C++ CPU; gate có thể xuống safety island |
| **Mô hình tiến trình** | thư viện trong process / dịch vụ riêng qua IPC | t_IPC, cô lập lỗi | an toàn: tách process theo mức ASIL; hiệu năng: cùng process |
| **Cơ chế IPC** | shared memory zero-copy (iceoryx), DDS/ROS 2, SOME/IP, socket | t_IPC p99, copy/frame | frame đi shared memory; metadata đi DDS |
| **Hệ điều hành / hypervisor** | Linux PREEMPT_RT, QNX, phân vùng VM (QC Linux primary + guest) | latency ngắt, jitter cyclictest | SA8797P: QC Linux / Android guest / QNX add-on (docs/DEPLOY_SA8797P.md) |
| **Mixed criticality** | ASIL decomposition: perception QM, gate+controller ASIL cao, giám sát độc lập | FMEA, FFI (freedom from interference) | lý do gate phải đơn giản, không ML |
| **Đồng bộ thời gian** | timestamp tại cảm biến, PTP/gPTP, một đồng hồ tham chiếu | sai lệch clock, age đúng | age tính sai nếu capture và decide dùng hai clock |
| **Giám sát** | watchdog, heartbeat, challenge-response cho NPU | thời gian phát hiện lỗi | NPU treo phải được phát hiện < 1 chu kỳ gate |
| **Vòng đời model** | manifest, ký số, OTA, A/B rollback | B9 | |

---

## 4. Realtime với mô hình CNN / RNN / DNN

### 4.1 Khái niệm và ký hiệu

```
Tác vụ τ_i:  C_i  = thời gian thực thi xấu nhất (WCET)       T_i = chu kỳ       D_i = deadline tương đối (thường D ≤ T)
             R_i  = thời gian phản hồi xấu nhất (WCRT)          J_i = jitter kích hoạt / hoàn thành
             U    = Σ C_i / T_i   (hệ số sử dụng)              B_i = thời gian bị chặn (blocking) bởi tài nguyên không preempt
Deadline miss khi R_i > D_i.    Tardiness = max(0, R_i − D_i).
Tuổi dữ liệu (data age) = t_dùng − t_lấy_mẫu ;  Reaction time = t_đầu_ra_đầu_tiên_phản_ánh_sự_kiện − t_sự_kiện
```

Hàm lợi ích (utility) theo thời gian hoàn thành là cách ngắn nhất phân biệt ba loại realtime:

```
Hard :  u(t) = 1 nếu t ≤ D,  −∞ (lỗi hệ thống) nếu t > D
Firm :  u(t) = 1 nếu t ≤ D,  0 nếu t > D          (kết quả trễ vô dụng nhưng không nguy hiểm; được phép trễ một tỉ lệ nhỏ)
Soft :  u(t) = 1 nếu t ≤ D,  giảm dần > D         (kết quả trễ vẫn có ích, chất lượng giảm)
```

**Mô hình CNN/DNN có hai tính chất thuận lợi cho realtime** mà LLM không có: (1) thời gian thực thi **gần như hằng
số** theo đầu vào khi shape tĩnh (không có vòng lặp phụ thuộc dữ liệu, không sinh token); (2) đồ thị tính toán biết
trước → bộ nhớ lập kế hoạch tĩnh. Những gì phá vỡ tính hằng số: shape động, NMS/top-k phụ thuộc số object, tracker có
số track biến thiên, cache miss, DVFS/thermal, tranh chấp NPU, page fault, GC/Python. **RNN** thêm trạng thái giữa
các bước (phải reset và bảo vệ) nhưng mỗi bước vẫn là compute cố định.

### 4.2 Ba loại realtime: vấn đề phải nắm khi triển khai

#### Hard realtime (ví dụ: safety gate + controller, vòng điều khiển phanh)

Trễ một lần = lỗi an toàn. Phải chứng minh `R_i ≤ D_i` cho **mọi** trường hợp, không chỉ thống kê.

Những điều phải nắm:

1. **WCET phải xác định được**: shape tĩnh; không cấp phát động; không khóa chia sẻ với tác vụ ưu tiên thấp (hoặc
   dùng priority inheritance / ceiling); không I/O chặn; không Python/GC trên đường này. Đo WCET bằng
   `max` trên hàng chục nghìn lần chạy ở trạng thái nóng + margin (×1.2–1.5) hoặc phân tích tĩnh. Phần ML (CNN) rất
   khó đạt chuẩn hard trên NPU vì firmware/driver không cho phân tích WCET → **không đặt mạng neural trên đường
   hard**; đặt logic deterministic (gate) ở đó và coi đầu ra mạng là **đầu vào có thể vắng**.
2. **Accelerator là tài nguyên không preempt**: một job NPU đang chạy không bị ngắt → tác vụ ưu tiên cao có thể bị
   chặn `B = max C_j` của job ưu tiên thấp (mục 4.3). Hậu quả: VLM dài 900 ms trên cùng NPU với YOLO sẽ chặn YOLO
   gần một giây → phải tách đơn vị tính toán hoặc chia nhỏ job.
3. **Ngắt, lập lịch OS**: Linux chuẩn có latency ngắt hàng trăm µs–ms; cần PREEMPT_RT hoặc RTOS (QNX), `SCHED_FIFO`,
   CPU cô lập, IRQ affinity, `mlockall`, tắt power-saving C-state sâu.
4. **Nhiệt / DVFS**: WCET phải đo ở clock **thấp nhất có thể xảy ra** (sau throttle) hoặc khóa clock.
5. **Giám sát độc lập**: watchdog ngoài (safety island/MCU) phát hiện tác vụ hard treo trong ≤ 1 chu kỳ; có hành
   động an toàn (minimal risk maneuver).
6. **Chứng cứ**: golden vectors (ADAS_VLA 206 kịch bản) + test tương đương bản C++; traceability yêu cầu → test
   (ISO 26262 phần 6).

#### Firm realtime (ví dụ: perception mỗi frame 33 ms – detector, lane, tracker, TTC)

Kết quả trễ **bị bỏ** (frame sau đã đến), nhưng không nguy hiểm **nếu** tỉ lệ bỏ nhỏ và tầng sau chịu được khoảng
trống.

Những điều phải nắm:

1. **Chính sách khi trễ**: bỏ frame cũ, lấy frame mới nhất (ADAS_VLA `_AsyncVLMWorker` làm đúng điều này cho VLM;
   perception hiện chạy đồng bộ). Không bao giờ xếp hàng frame → latency tích luỹ (queue growth) là lỗi điển hình.
2. **Ràng buộc (m, k)-firm**: trong **mọi** k job liên tiếp có ít nhất m job kịp deadline. Ví dụ (m,k) = (9,10) cho
   perception: không được mất 2 frame liên tiếp ở 30 fps vì tracker/TTC cần dt ≤ 100 ms để ước lượng closing speed.
   Hold time của AEB/FCW (`aeb_hold_s` 0.5 s) là cơ chế **chịu khoảng trống** của tầng hard khi tầng firm trễ; cửa sổ
   xác nhận `aeb_confirm_s` 0.1 s (đếm xuyên qua các lần mất phát hiện ≤ `aeb_confirm_gap_s` 0.1 s) là bộ lọc kiểu
   (m,k) ở chiều ngược lại: AEB chỉ tác động khi trigger giữ đủ lâu, đổi 100 ms thời gian phản ứng lấy ít AEB ảo hơn
   (đo trên PC 29/09: AEB sai trên clip bình thường 15,8 % → 4,0 % thời gian). 100 ms này **phải được cộng vào**
   ngân sách reaction time của B0 (xem ví dụ mục 4.3).
3. **Thước đo**: DMR, chuỗi miss dài nhất, age p99. Mục tiêu đề xuất: DMR ≤ 1 %, không có 2 miss liên tiếp,
   age p99 ≤ 2T.
4. **Thiết kế tránh miss**: WCET perception ≤ 0.7T (headroom 30 %); hậu xử lý có giới hạn trên (top-k, max track);
   lane detection có thể chạy ở T' = 2T nếu ngân sách thiếu (lane đổi chậm hơn object).
5. **Tính xác định của kết quả**: cùng input → cùng output để replay được (ADAS_VLA dùng timeline video, không dùng
   wall-clock khi offline).

#### Soft realtime (ví dụ: VLM meta-action, HUD, LLM giải thích)

Kết quả trễ vẫn hữu ích, giá trị giảm dần; đo bằng **phân phối** (p50/p95/p99), không phải WCET.

Những điều phải nắm:

1. **Ngân sách thống kê** thay cho deadline cứng: p50 ≤ 1 s, p95 ≤ 1.5 s (ADAS_VLA mục tiêu p50 ≤ 1 s; đã đo 0.90 s
   trên RTX 4060, chưa đo trên HTP).
2. **Tuổi tối đa và fallback**: kết quả soft chỉ được tầng hard dùng khi `age ≤ max_decision_age_s` (2 s); quá tuổi
   → luật. Đây là ranh giới soft → hard và là **một test** (fault injection B8).
3. **Giảm chất lượng có kiểm soát** (graceful degradation): rút gọn prompt, bỏ `reason`, giảm tần suất gọi khi
   nhiệt cao; hiển thị HUD có thể giảm fps.
4. **Không để soft phá hard**: soft task phải ở ưu tiên thấp, core/accelerator khác hoặc có job đủ nhỏ để
   blocking `B` chấp nhận được (mục 4.3); bộ nhớ của nó không được làm swap/compact ảnh hưởng hard task.
5. **Đo tác động**: với VLM, thước đo cuối không phải latency mà là **under-braking sau gate** – soft task chậm
   nhưng đúng vẫn tốt hơn nhanh mà sai; latency chỉ cần đủ để age ≤ ngân sách.

### 4.3 Deadline và scheduling: công thức cần dùng

**Mô hình tác vụ chu kỳ** (Liu & Layland): mỗi tác vụ (C_i, T_i, D_i), độc lập, preempt được, một core.

```
Rate Monotonic (RM, ưu tiên theo chu kỳ ngắn nhất), D_i = T_i:
    đủ điều kiện lập lịch được nếu  U = Σ C_i/T_i ≤ n·(2^{1/n} − 1)     (n→∞: 0.693)
Earliest Deadline First (EDF), D_i = T_i:
    lập lịch được  ⇔  U ≤ 1
Phân tích thời gian phản hồi (RTA, điều kiện cần & đủ cho ưu tiên cố định):
    R_i^{(0)} = C_i
    R_i^{(k+1)} = C_i + B_i + Σ_{j ∈ hp(i)} ⌈ R_i^{(k)} / T_j ⌉ · C_j      lặp đến hội tụ; cần R_i ≤ D_i
Blocking bởi tài nguyên không preempt (NPU, GPU queue, mutex không có ceiling):
    B_i = max_{j ∈ lp(i)} C_j^{res}    (job dài nhất của tác vụ ưu tiên thấp hơn trên cùng tài nguyên)
```

Hệ quả thực tế cho ADAS_VLA: nếu VLM (C ≈ 900 ms) và YOLO (T = 33 ms) dùng **cùng** NPU không preempt thì
`B_YOLO = 900 ms ≫ D_YOLO` → không lập lịch được. Giải pháp: NPU/đơn vị khác, chia job VLM thành các chunk nhỏ
(prefill theo lớp), hoặc chạy VLM trên GPU/CPU với ưu tiên thấp. **Đây là điểm phải xác nhận với Qualcomm**
(số core HTP, có chia sẻ theo thời gian hay không).

**Chuỗi nhân-quả nhiều tốc độ** (camera 30 Hz → perception 30 Hz → gate 50–100 Hz → actuator), giao tiếp bằng
"register" (ghi mới nhất, đọc khi cần):

```
Latency tốt nhất:        L_min = Σ C_i
Giới hạn trên tuổi dữ liệu / reaction time (Davare et al. 2007, hệ thống không đồng bộ):
                         L_max ≤ Σ_i ( T_i + R_i )
Với kích hoạt đồng bộ theo pha (time-triggered, LET): L = Σ T_i  (xác định, không phụ thuộc R_i miễn R_i ≤ T_i)
```

Ví dụ ADAS_VLA (mục tiêu đề xuất): camera 33 ms + perception (T 33, R ≤ 25) + gate (T 10, R ≤ 1) + controller
(cùng chu kỳ gate) → `L_max ≈ 33 + (33+25) + (10+1) ≈ 102 ms` chưa tính độ trễ phanh thuỷ lực. Cộng thêm cửa sổ
xác nhận AEB `aeb_confirm_s` 0.1 s (mục 4.2) và cửa sổ hồi quy vận tốc `velocity_window_s` 0.5 s (TTC chỉ ổn định khi
đã có đủ mẫu) thì thời gian từ lúc hazard xuất hiện đến lệnh phanh là ≈ 200 ms cộng thời gian TTC hội tụ. Vượt ngân
sách AEB 100–150 ms đề xuất ở B0 → B0 phải quyết định: chấp nhận (đánh đổi với AEB ảo của camera đơn) hay giảm bằng
perception tốt hơn (fusion radar/depth). Mọi tối ưu dataflow (3.1) ở đây có ý nghĩa an toàn, không chỉ hiệu năng.

**Jitter** và vì sao nó tệ hơn latency: bộ ước lượng closing speed/TTC hồi quy khoảng cách theo thời gian trong cửa sổ
`velocity_window_s` 0.5 s (trước là hiệu hai frame liên tiếp: nhiễu vài % khoảng cách ở 60 fps thành hàng chục m/s, một
nguyên nhân của AEB ảo đã sửa 29/09); hồi quy chịu nhiễu khoảng cách tốt hơn nhưng vẫn cần trục thời gian đúng: jitter
ở timestamp làm độ dốc sai → TTC sai. Dùng **timestamp capture** của cảm biến, không dùng thời điểm xử lý.

**Cấu hình OS tối thiểu cho tác vụ hard/firm trên Linux**: kernel PREEMPT_RT; `SCHED_FIFO` ưu tiên gate > perception
> VLM > HUD; `isolcpus` + `taskset` cho gate; IRQ affinity camera về core perception; `mlockall(MCL_CURRENT|MCL_FUTURE)`;
tắt C-state sâu / khóa clock; đo nền bằng `cyclictest` (latency ngắt p99.99 phải < 100 µs trước khi đo model).

### 4.4 Thước đo realtime và tiêu chí chấp nhận theo loại

| Thước đo | Công thức | Hard (gate, controller) | Firm (perception) | Soft (VLM, HUD) |
|---|---|---|---|---|
| WCET quan sát | `max` trên ≥ 10⁴ lần, nóng | ≤ 0.5·D (headroom 50 %) | ≤ 0.7·D | không bắt buộc |
| Deadline-miss ratio | miss / jobs | **0** | ≤ 1 %, không 2 miss liên tiếp | p95 ≤ ngân sách |
| Jitter | p99 − p50 của thời điểm hoàn thành | ≤ 10 % D | ≤ 20 % T | – |
| Tuổi dữ liệu p99 | `t_dùng − t_capture` | n/a (dùng dữ liệu firm) | ≤ 2T | ≤ `max_decision_age_s` |
| Reaction time | mục 4.1 | ≤ ngân sách AEB (B0) | – | – |
| Số allocation/frame | hook malloc | 0 | 0 | tuỳ |
| Page fault/s sau khởi động | `/proc/<pid>/stat` | 0 | 0 | – |
| Latency sau 30 phút / lạnh | tỉ số | ≤ 1.05 | ≤ 1.1 | ≤ 1.2 |

Cách đo đúng: tracing có timestamp ở ranh giới mỗi tầng (ftrace/LTTng trên Linux, QNN profiler cho NPU), histogram
độ phân giải cao (HDR) thay vì mean/std; chạy đồng thời **toàn bộ** tải (perception + VLM + HUD + log) khi đo một
tầng, vì tranh chấp mới là nguồn jitter chính.

---

## 5. Áp dụng: ADAS_VLA và ADAS_CNN

### 5.1 ADAS_VLA: bảng tác vụ và ngân sách (đề xuất cho B0)

| Tác vụ | Mô-đun hiện tại | Loại RT | T | D (đề xuất) | Nơi chạy đích | Chỉ tiêu accuracy |
|---|---|---|---|---|---|---|
| Thu ảnh + letterbox | `sources.py`, `deploy/export.py` | firm | 33 ms | 5 ms | ISP / CPU, zero-copy | – |
| Detector YOLO11s 384×640 INT8 | `perception/detector.py` | firm | 33 ms | 15 ms | HTP | mAP ≥ FP32 − 1 điểm; recall xe gần ≥ FP32 |
| Lane YOLOP 640×640 | `perception/lanes.py` | firm | 33–66 ms | 10 ms | HTP (hoặc T = 2 frame) | offset_norm sai số ≤ 0.1 |
| Tracker ByteTrack + geometry/TTC/cut-in | `perception/geometry.py` | firm | 33 ms | 3 ms | CPU C++ | TTC sai số theo golden perception |
| **Safety gate** | `control/safety.py` → C++ | **hard** | 10 ms | 1 ms | CPU cô lập / safety island | 206/206 golden vectors |
| **Controller P** | `control/controller.py` → C++ | **hard** | 10 ms | 0.5 ms | cùng gate | golden vectors |
| VLM Qwen2.5-VL-3B W4A16, 220 token, 2 frame | `reasoning/vlm.py` | soft | sự kiện / 15 frame | p50 ≤ 1 s, age ≤ 2 s | HTP (ViT + LM) hoặc GPU | under-braking sau gate ≤ 1 %, joint ≥ 85 % |
| HUD / log | `hud.py`, `events.py` | soft | 33 ms | best-effort | GPU/CPU | – |
| LLM giải thích Qwen3-4B | `reasoning/llm.py` | best-effort | theo yêu cầu | – | HTP (Genie) | – |

Kiểm tra lập lịch sơ bộ cho core CPU dành riêng (gate + controller + tracker/geometry, RM, D = T):

```
U = 1/10 + 0.5/10 + 3/33 ≈ 0.10 + 0.05 + 0.09 = 0.24  ≤ 0.78 (n = 3)  → lập lịch được với dư lớn
RTA gate (ưu tiên cao nhất, B = 0 nếu không khoá chung với tác vụ khác): R = C = 1 ms ≤ 10 ms
```

Chuỗi camera → phanh: xem ví dụ mục 4.3 (`L_max ≈ 102 ms`). Hai rủi ro lập lịch cần giải quyết **trước** khi lên
board: (1) VLM và CNN cùng NPU không preempt (mục 4.3); (2) Python trên đường firm/hard (chỉ chấp nhận trên PC).

### 5.2 ADAS_VLA: ánh xạ các bước B0–B9 vào repo

| Bước | Có gì trong repo | Còn thiếu |
|---|---|---|
| B0 | ngân sách VLM (p50 ≤ 1 s), chỉ tiêu under-braking ≤ 1 %, joint ≥ 85 % (CLAUDE.md) | bảng 5.1 cho perception/gate chưa được đo/chốt |
| B1 | val 900 mẫu cố định, Wilson CI trong `evaluate.py`, report HTML | golden outputs của detector/lane (logits) chưa lưu |
| B2 | `adas-vla export` → ONNX tĩnh `[1,3,384,640]`, NMS ngoài graph, calibration raw | parity check ONNX ↔ PyTorch cho YOLO chưa tự động; ViT 2 frame chưa export |
| B3 | – | chỉ làm nếu B7 thiếu ngân sách |
| B4 | calibration set từ video mẫu | encodings QAIRT, bảng FP32 vs INT8 cho detector; AIMET cho VLM |
| B5 | hướng dẫn qairt-converter/quantizer (docs/DEPLOY_SA8797P.md) | chưa chạy trên SDK thật |
| B6 | pipeline dual-rate, `FrameHistory`, `max_decision_age_s`, `_AsyncVLMWorker` | bản C++ cho tracker/geometry/gate/controller; zero-copy |
| B7 | đo p50 VLM trên RTX 4060 | mọi số trên HTP |
| B8 | 65 unit test, 206 golden vectors, `safety_golden.py --check`, `gate_offline` trong eval, `scripts/gate_replay.py` (replay SIL: 111 video Nexar + 24 clip Australian + clip cao tốc; kết quả 29/09 trong CLAUDE.md) | test tương đương C++; số liệu replay theo giờ (lần mỗi giờ); fault injection có hệ thống (VLM trễ/vắng, mất frame) |
| B9 | – | manifest, watchdog, telemetry |

### 5.3 ADAS_CNN: ánh xạ milestone ↔ bước

| Milestone ADAS_CNN | Bước ở đây | Thước đo đã có trong repo |
|---|---|---|
| M3 baseline FP32 | B1 | accuracy / mAP@0.5, split theo track/drive, `experiments/results/*_fp32_torch.json` |
| M4 trade-off kiến trúc / độ phân giải | B3 | ≥ 3 dòng FP32 + latency host |
| M5 export ONNX & parity | B2 | `parity_ok` (max abs diff < 1e-3), op histogram |
| M6 quantization | B4 | bảng FP32 / INT8-PTQ (≥ 2 calibration) / QAT |
| M7 deploy Qualcomm | B5 + B7 | accuracy board vs ORT INT8 ≤ 1 %, `*_htp.json`, `board_env_*.txt` |
| M8 benchmark | B7 | p50/p95/p99, model-only vs end-to-end, memory, power/thermal (`docs/09`) |

ADAS_CNN **chưa có** mục tương ứng B6/B8/B9 (tích hợp pipeline, kiểm chứng hệ thống, vận hành) vì là project học
perception đơn lẻ; phần realtime mục 4 áp dụng cho nó ở mức "firm task đơn": DMR, (m,k), jitter của một model
CenterNet/ResNet trên HTP. Khi ghép CenterNet KITTI vào vai detector của ADAS_VLA, ngân sách là dòng "Detector" ở 5.1.

### 5.4 Gợi ý thứ tự làm (không đổi ưu tiên đã ghi trong CLAUDE.md)

1. Chốt bảng 5.1 (B0) và thêm `t_pre / t_model / t_post` + timestamp capture vào log per-frame của `adas-vla run`
   (B6) – không cần GPU, cho số nền trên PC.
2. Port C++ gate + controller với test golden (B8), đo WCET trên PC bằng `perf stat` / vòng lặp 10⁵ lần (B7 mức host).
3. Detector: parity ONNX (B2) → PTQ INT8 trên host bằng ORT (B4) → khi có SDK/board: B5, B7.
4. VLM theo checklist docs/DEPLOY_SA8797P.md mục 5, sau khi CNN đã ổn trên HTP.

---

## 6. Trạng thái số liệu và việc cần đo tiếp

| Số | Nguồn | Trạng thái |
|---|---|---|
| VLM v3 p50 0.90 s, joint 72.6 %, under-braking 8.8 %, JSON 100 % (ds_v2 val 900) | CLAUDE.md, eval v3 trên RTX 4060 | **đã đo** (PC, 28/09) |
| Nexar crash v3: 62.7 % / under-braking 4.0 % | CLAUDE.md | **đã đo** (PC, 28/09) |
| v3 vs v4 trên ds_v3 (val 900 / Nexar 424): v3 73.1 % joint / under 9.1 % / over 14.3 %; v4 67.1 % / 6.2 % / 23.7 % (Nexar: v3 63.7 / 3.5 / 32.8; v4 61.1 / 2.4 / 36.6) | CLAUDE.md | **đã đo** (PC, 30/09); v3 vẫn là mặc định |
| AEB sai trên clip lái bình thường: 15,8 % thời gian / 8,6 lần mỗi phút → 4,0 % / 2,3 lần mỗi phút; AEB trong cửa sổ nguy hiểm Nexar 53 % (đối chứng 25 %) | `scripts/gate_replay.py`, CLAUDE.md | **đã đo** (PC, 29/09) |
| YOLOP ≈ 7 ms trên GPU PC | ghi chú configs/default.yaml | ước tính trên PC, chưa có protocol |
| Mọi số latency / memory / power trên HTP, SA8797P | – | **chưa đo** |
| Ngân sách T, D trong bảng 5.1 | tài liệu này | **đề xuất**, cần chốt với chủ repo |
| WCET gate/controller C++ | – | chưa có bản C++ |

---

## 7. Checklist tổng hợp

Trước khi nói "đã deploy xong" một model CNN/DNN trên board:

- [ ] B0: bảng ngân sách (T, D, loại RT, nơi chạy, MB, chỉ tiêu accuracy) được chốt và lưu trong repo
- [ ] B1: val đóng băng, metric + CI + golden outputs có hash
- [ ] B2: shape tĩnh, `e_max < 1e-3`, 0 op không hỗ trợ, MACs/params in ra
- [ ] B4: calibration từ train, bảng FP32/INT8 (+CI), lớp giữ FP16 có lý do
- [ ] B5: 0 fallback CPU trên hot path, accuracy board vs host ≤ 1 điểm trên cùng subset
- [ ] B6: `t_pre/model/post` tách, 0 allocation trong vòng lặp, timestamp capture dùng cho age
- [ ] B7: p50/p95/p99/max sau warm-up, ≥ 3000 lần, ở trạng thái nóng 30 phút, cùng toàn bộ tải; roofline để biết
      bottleneck; J/inference nếu đo được
- [ ] B8: hard: 0 miss, golden 100 %; firm: DMR ≤ 1 %, không 2 miss liên tiếp; soft: age ≤ ngân sách và fallback
      hoạt động 100 % khi fault injection
- [ ] B9: manifest + telemetry (DMR, nhiệt, fallback) + rollback đã tập

---

## 8. Tài liệu tham khảo (công khai)

Lý thuyết realtime và lập lịch
- C. L. Liu, J. W. Layland, "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment", JACM 1973
  (RM bound, EDF).
- M. Joseph, P. Pandya, "Finding Response Times in a Real-Time System", 1986; N. Audsley et al., 1993 (RTA).
- G. Buttazzo, *Hard Real-Time Computing Systems*, Springer, 3rd ed. (tổng quan, blocking, (m,k)-firm).
- M. Hamdaoui, P. Ramanathan, "A Dynamic Priority Assignment Technique for Streams with (m,k)-Firm Deadlines", 1995.
- A. Davare et al., "Period Optimization for Hard Real-time Distributed Automotive Systems", DAC 2007 (giới hạn
  latency chuỗi nhiều tốc độ Σ(T_i + R_i)).
- Linux Foundation, "Real-Time Linux (PREEMPT_RT)" wiki; `cyclictest` (rt-tests).

Hiệu năng, roofline, benchmark
- S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model", CACM 2009.
- MLPerf Inference / MLPerf Tiny rules (warm-up, percentile, scenario single-stream vs offline).
- ADAS_CNN `docs/09_benchmark_protocol.md` (protocol của repo).

Lượng tử hóa và nén
- M. Nagel et al., "A White Paper on Neural Network Quantization", Qualcomm AI Research, 2021.
- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
  CVPR 2018.
- G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network", 2015.
- AIMET: https://github.com/quic/aimet ; ADAS_CNN `docs/06_quantization.md`, `docs/07_pruning_distillation_resolution.md`.

Toolchain Qualcomm (công khai)
- Qualcomm AI Engine Direct / QAIRT documentation (auto platform overview); Qualcomm AI Hub models/apps;
  Ultralytics QNN export; ONNX Runtime QNN Execution Provider – liên kết trong `docs/DEPLOY_SA8797P.md`.
- ADAS_CNN `docs/08_qualcomm_deployment.md`, `scripts/deploy/qualcomm/` (đánh dấu UNVERIFIED cho đến khi chạy trên SDK thật).

An toàn chức năng
- ISO 26262:2018 (Road vehicles – Functional safety), đặc biệt phần 6 (software) và khái niệm ASIL decomposition,
  freedom from interference; ISO 21448:2022 (SOTIF).
