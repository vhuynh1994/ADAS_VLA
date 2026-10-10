# ADAS-VLA

Pipeline **Vision-Language-Action** cho ADAS, chạy trên PC (RTX 4060 Laptop 8 GB) và thiết kế để
deploy dần lên **Snapdragon Ride Elite (SA8797P)**. Xem [docs/DEPLOY_SA8797P.md](docs/DEPLOY_SA8797P.md).

```
 camera ─┬─► YOLO11 + ByteTrack ─► distance / TTC / in-path ─┐        (mọi frame, ~ms)
         └─► lane detection (OpenCV) ────────────────────────┤
                                                              ▼
                                                        SceneContext ──► summary_text()
                                                              │                 │
                          (mỗi N frame, sync hoặc async)       │                 ▼
                           frame (ảnh cố định 560×308) ───────┼──────► VLM (Qwen2.5-VL) ─► JSON meta-action
                                                              ▼                 │         (recommendation)
                                              SAFETY GATE (deterministic) ◄─────┘
                          rules/ACC fallback · AEB · FCW · VRU · speed cap · lane-change confirm · LDW
                                                              ▼
                                         Controller (P speed + lane keeping) ─► throttle / brake / steer
```

**Nguyên tắc:** VLM chỉ *khuyến nghị*. Safety gate deterministic mới là bên *quyết định* (tầng cuối phải
kiểm chứng được, không giao cho mạng neural). Nếu VLM chậm, lỗi JSON hoặc quá hạn (`max_decision_age_s`), hệ thống
tự chuyển sang policy ACC dựa trên luật.

## Cài đặt

```bash
cd ~/workspaces/adas-vla
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
./scripts/download_samples.sh          # video dashcam mẫu (Udacity, MIT) + ảnh đường phố
pytest -q                              # unit test, không cần GPU/model
```

Weights YOLO (`yolo11n.pt`) được tự tải trong lần chạy đầu. Với VLM/LLM, nên tải trước bằng script
(có resume, tự retry, kiểm tra sha256; phù hợp mạng chậm/chập chờn). Model được lưu vào `models/` và tự được nhận:

```bash
./scripts/download_models.sh smoke     # ~1.5 GB: SmolVLM-256M + Qwen2.5-0.5B để chạy thử nhanh
./scripts/download_models.sh           # + Qwen2.5-VL-3B (7.5 GB) + Qwen3-4B-Instruct-2507 (8 GB)
```

## Sử dụng

```bash
# 1) Chạy trên video, xuất video có HUD + log quyết định từng frame
adas-vla run --source data/samples/highway_traffic.mp4 --output outputs/demo.mp4 --log outputs/demo.jsonl

# chỉ perception + rules (không load VLM), để test nhanh
adas-vla run --source data/samples/highway_traffic.mp4 --no-vlm --output outputs/rules.mp4

# camera trực tiếp: VLM chạy thread nền
adas-vla run --source 0 --show --vlm-mode async

# 2) Phân tích 1 ảnh → in JSON quyết định
adas-vla analyze --image data/samples/street.jpg --ego-speed 30

# 3) Copilot hỏi đáp về cảnh (tiếng Việt)
adas-vla chat --image data/samples/highway_traffic.mp4 --frame 600 --set vlm.language=vi

# 4) LLM giải thích các lần hệ thống can thiệp (AEB/FCW/PED/LDW) từ log của bước 1
adas-vla explain --log outputs/demo.jsonl

# Chạy thử nhanh bằng model nhỏ (sau khi `download_models.sh smoke`)
adas-vla analyze --config configs/smoke.yaml --image data/samples/street.jpg

# Profile nhanh cho PC (Qwen3-VL-2B bf16, Apache-2.0)
adas-vla run --config configs/pc_fast.yaml --source data/samples/highway_short.mp4 --output outputs/fast.mp4
```

Có thể override mọi tham số: `--set control.cruise_speed_kmh=80 --set vlm.every_n_frames=10`.

## Quyết định của VLM (2 trục)

VLM trả về JSON ngắn (~40 token), action luôn đứng đầu (assistant được pre-fill `{"longitudinal":`):

```json
{"longitudinal": "KEEP", "lateral": "KEEP_LANE", "target_speed_kmh": 60, "risk": "low", "reason": "Lane ahead is clear; hold speed."}
```

- **longitudinal**: `KEEP` · `ACCELERATE` · `DECELERATE` · `BRAKE` · `STOP` (`EMERGENCY_BRAKE` chỉ safety gate được phát)
- **lateral**: `KEEP_LANE` · `NUDGE_LEFT` · `NUDGE_RIGHT` · `CHANGE_LEFT` · `CHANGE_RIGHT`

## Quy trình dữ liệu → fine-tune → đánh giá

```bash
# 1. Tải dữ liệu dashcam công khai (resume + kiểm tra sha256)
python scripts/fetch_hf.py --dataset commaai/comma2k19 --include "raw_data/Chunk_1.zip" --out data/raw
python scripts/fetch_hf.py --dataset qutegocentric/Australian_Roads_Dashcam_Driving --include "*.mp4" --include "*.csv" --out data/raw
python scripts/fetch_hf.py Qwen/Qwen2.5-VL-7B-Instruct            # teacher gán nhãn cho clip không có CAN

# 2. Tạo dataset (train/val tách theo video, không bao giờ trộn frame của cùng video)
adas-vla build-dataset comma2k19 --out data/ds_v1 --segments 100                    # nhãn từ CAN (hành vi tài xế thật)
adas-vla build-dataset australian --out data/ds_v1 --config configs/teacher_7b.yaml  # near-crash: teacher 7B + safety gate
adas-vla build-dataset uk --out data/ds_v1 --config configs/teacher_7b.yaml

# 3. Review nhãn trên web (phím 1-5 / Q W E A D / Enter). Thứ tự: val trước, nhãn bị gắn cờ, nhãn hiếm
adas-vla review --data data/ds_v1/labels.jsonl        # mở http://127.0.0.1:8765

# 4. QLoRA fine-tune trên split train, cân bằng lớp (chỉ LM; ViT đóng băng → sub-model ViT trên NPU không đổi)
adas-vla train --data data/ds_v1/labels.jsonl --output checkpoints/lora-v1 --epochs 2

# 5. Đánh giá trên split val (--reviewed-only: chỉ mẫu đã được người review)
adas-vla eval --data data/ds_v1/labels.jsonl --split val --report outputs/eval_base.jsonl
adas-vla eval --data data/ds_v1/labels.jsonl --split val --adapter checkpoints/lora-v1 --report outputs/eval_v1.jsonl
adas-vla report --run base=outputs/eval_base.jsonl --run lora-v1=outputs/eval_v1.jsonl --data-dir data/ds_v1

# 6. Gộp LoRA vào weights (bỏ overhead adapter khi chạy) rồi dùng
adas-vla merge --adapter checkpoints/lora-v1 --out models/adas-vlm-v1
adas-vla run --source ... --set vlm.model_id=models/adas-vlm-v1
```

Định dạng dataset: xem `adas_vla/training/data.py`. Review được lưu ở `labels.reviews.jsonl` (không sửa nhãn gốc).

## Export cho Qualcomm

```bash
adas-vla export --out outputs/deploy            # YOLO → ONNX static [1,3,384,640] + calib/*.raw + input_list.txt
adas-vla export --out outputs/deploy --qnn-arch 79   # thêm bản Ultralytics QNN (onnxruntime-qnn)
```

## Cấu trúc

```
adas_vla/
  types.py            Detection, LaneInfo, SceneContext, DrivingDecision, ControlCommand, Alert
  config.py           dataclass config + YAML + --set overrides
  perception/         detector.py (YOLO11s+ByteTrack, màu đèn), geometry.py (distance/TTC/oncoming), lanes.py (YOLOP CNN hoặc classic),
                      depth.py (Depth-Anything thay khoảng cách pinhole, căn scale theo mặt đường; tắt mặc định)
  reasoning/          prompts.py, parser.py (JSON chịu lỗi), vlm.py (transformers, 4-bit, LoRA)
  control/            safety.py (safety gate, AEB/FCW hysteresis), controller.py, golden.py (đọc/replay golden vectors)
  pipeline.py         dual-rate pipeline, VLM sync/async
  hud.py  sources.py  cli.py
  training/           autolabel.py, finetune.py (QLoRA + cân bằng lớp), evaluate.py, data.py
  datasets/           comma2k19.py (nhãn từ CAN), clips.py (teacher + safety), common.py
  review.py report.py trang review nhãn · báo cáo HTML so sánh các lần đánh giá
  deploy/export.py    ONNX + calibration cho QAIRT
  reasoning/llm.py    LLM text-only giải thích sự kiện ADAS · events.py: tách sự kiện từ log
configs/              default.yaml (Qwen2.5-VL-3B + Qwen3-4B, 4-bit) · smoke.yaml · pc_fast.yaml · teacher_7b.yaml
scripts/              download_samples.sh · download_models.sh · fetch_hf.py · sweep_policy.py · safety_golden.py
tests/data/safety_golden.json   kịch bản → quyết định mong đợi của safety gate (test tương đương cho bản C++)
.github/workflows/ci.yml        pytest trên Python 3.10 + 3.12, không cần GPU
docs/DEPLOY_SA8797P.md          đường deploy lên Snapdragon Ride Elite (toolchain công khai)
docs/AI_DEPLOYMENT.md           quy trình deploy B0–B9, DoD + công thức đo, phân loại tối ưu, realtime hard/firm/soft
```

## Kết quả đã kiểm chứng trên PC này (RTX 4060 Laptop 8 GB, 27/09/2026)

| Thành phần | Kết quả |
|---|---|
| Unit test | 44/44 pass (`pytest -q`); 61/61 sau phiên 29/09 (chạy trên cloud, Python 3.11, không GPU) |
| Perception + safety + HUD (YOLO11n, ByteTrack, lane) | ~12 ms/frame; 22 FPS kể cả ghi video |
| VLM Qwen2.5-VL-3B 4-bit, 560×308 | JSON hợp lệ 15/15, latency p50 2.1 s, **VRAM đỉnh 2.7 GB** (gồm cả YOLO) |
| Chế độ async | vòng perception giữ ~30 FPS trong khi VLM chạy nền |
| Safety gate | ảnh đường phố: VLM đề xuất `NUDGE_LEFT`, gate ghi đè thành `EMERGENCY_BRAKE` (người trong làn, 3 m) |
| Copilot chat (tiếng Việt) | 0.6–1.5 s/câu; được đưa cảnh báo an toàn vào ngữ cảnh nên không phủ nhận AEB |
| QLoRA fine-tune (12 mẫu, 4 epoch) | 80 s, VRAM đỉnh 4.7 GB, loss 0.29 → 0.085 |
| Eval trên tập train (smoke test, chưa phải generalization) | action accuracy 41.7% → 100%, under-braking 8.3% → 0% |
| LLM explain Qwen3-4B 4-bit | tiếng Việt chính xác, ~2 s/sự kiện |
| Export YOLO → ONNX static + 100 mẫu calibration | OK |

Nhận xét: nếu không fine-tune, VLM mô tả cảnh tốt nhưng chọn action thiếu nhất quán (hay chọn `NUDGE_LEFT` trên
làn trống). Vì vậy bước **label → review → LoRA** là bắt buộc trước khi đánh giá nghiêm túc.

## Kết quả fine-tune VLM (3 vòng, 27/09/2026)

Đo trên **val 900 mẫu cố định** (nguyên tuyến đường / clip chưa từng thấy khi train) và **test va chạm Nexar**
(424 mẫu). Báo cáo đầy đủ: `outputs/report_val.html`, `outputs/report_nexar.html`. Dữ liệu: [docs/DATASET.md](docs/DATASET.md).

| | Zero-shot | v1 | v2 | **v3 (mặc định)** | Mục tiêu |
|---|---|---|---|---|---|
| Joint accuracy (val) | 39.4% | 61.1% | 66.3% | **72.6%** | ≥ 85% ✗ |
| Under-braking (val) | 18.0% | 6.3% | 7.3% | 8.8% | ≤ 1% ✗ |
| Valid JSON | 100% | 100% | 100% | **100%** | 100% ✓ |
| Latency p50 (RTX 4060) | 1.20 s | 2.10 s | 0.96 s | **0.90 s** | ≤ 1 s ✓ |
| Nexar crash test: accuracy / under-braking | 25.5% / 47.9% | — | 16.5% / 37.5% | **62.7% / 4.0%** | |

- v3 = warm start từ v2, 12.461 mẫu train (comma2k19 CAN + clip Úc + Nexar), `models/adas-vlm-v3` (LoRA đã merge).
- **Cautious decoding** (`vlm.action_policy: cautious`) đã thử: muốn under-braking ≈ 1–3% thì phải phanh thừa ở
  ~40% cảnh bình thường → giữ `greedy`. Lỗi thiếu phanh còn lại là model *tự tin sai* (không thấy nguy hiểm),
  không phải do ngưỡng.
- Nguyên nhân chính chưa đạt: chỉ nhìn **1 frame** (không thấy xe trước đang chậm dần / xe sắp cắt làn),
  perception bỏ sót vật thể quá gần, nhãn near-crash mơ hồ, lane change hiếm (model không dự đoán CHANGE_*).
- **Safety gate** vẫn phanh trong các ca VLM bỏ sót (xem `outputs/demo_crash_v3_h264.mp4`).

## Cập nhật 29/09/2026 (viết trong phiên cloud, đã kiểm chứng trên GPU — kết quả ở mục sau)

- **Perception:** bbox chạm mép dưới khung hình (xe rất gần bị cắt) không còn bị ước lượng *xa hơn thật*: dùng thêm
  chiều rộng bbox và mặt đường (`camera.mount_height_m`) làm cận trên. Xe làn bên đang **cắt làn** được gắn cờ
  `cutting_in` (`CUTTING IN from the left/right` trong context của VLM, nhãn `cut-in` trên HUD) và được safety gate
  coi là lead → ACC/FCW phản ứng trước khi nó vào hẳn làn. Tham số: `perception.cut_in_rate`, `cut_in_max_distance_m`.
- **Safety gate:** AEB/FCW có hysteresis và thời gian giữ (`safety.aeb_hold_s`, `fcw_hold_s`, `hysteresis`) — một
  frame nhiễu không còn làm phanh nhấp nhả. **Golden vectors** `tests/data/safety_golden.json` (206 kịch bản, sinh bằng
  `python scripts/safety_golden.py`) là test tương đương cho bản C++ trên SA8797P; CI kiểm tra file luôn khớp với gate.
- **VLM 2 frame:** `vlm.prev_frame_s: 0.5` đưa thêm frame 0,5 s trước dưới dạng video 2 frame. Qwen2.5-VL gói 2 frame
  vào một temporal patch nên **vẫn 220 visual token**, latency gần như không đổi, model nhìn được xe trước chậm dần /
  xe cắt làn. Dataset builder lưu `image_prev`; mẫu cũ không có thì dùng frame hiện tại 2 lần (Qwen mã hóa ảnh đơn
  đúng như vậy). Cần fine-tune v4 với cùng cấu hình; v3 giữ `prev_frame_s: 0`.
- **Policy `cautious_gated`:** chỉ escalate DECELERATE/BRAKE theo xác suất khi perception xác nhận có mối nguy trong
  40 m (`types.context_has_hazard_cue`, cùng một luật cho online và offline). Sweep offline từ eval JSONL sẵn có:
  `python scripts/sweep_policy.py outputs/eval_v3.jsonl --data data/ds_v2/labels.jsonl --gate`.
- **Train:** `--workers N` (DataLoader worker giải mã ảnh + tokenize song song với GPU), `--brake-weight 2.0`
  (nhân loss của mẫu DECELERATE/BRAKE/STOP, nhắm thẳng under-braking).
- **Eval/report:** khoảng tin cậy Wilson 95% cho mọi tỷ lệ; thêm **under-braking sau safety gate** (VLM + gate = hệ
  thống) khi dataset có trường `lead` (builder mới ghi; ds_v2 cũ phải build lại mới có).
- **CI:** `.github/workflows/ci.yml` chạy pytest với Python 3.10 và 3.12 (không cần torch), kiểm tra cú pháp mọi module
  và golden vectors. Sửa lỗi f-string lồng nhau trong `reasoning/llm.py` chỉ chạy được trên Python ≥ 3.12.

## Kiểm chứng trên PC (29/09/2026)

Hold/hysteresis của AEB kéo dài các phát hiện sai của perception đơn camera thành phanh khẩn cấp nhiều giây
(clip bám xe bình thường: AEB 33% thời gian). Đã sửa tận gốc ở perception và thêm xác nhận mục tiêu cho AEB:

- **Vận tốc tiếp cận / TTC** (`perception/geometry.py`): trước đây lấy hiệu khoảng cách giữa 2 frame liên tiếp → ở 60 fps
  nhiễu vài % của khoảng cách thành hàng chục m/s ("xe cách 24 m lao tới 59 km/h"). Giờ là hồi quy tuyến tính khoảng cách
  theo thời gian trong `perception.velocity_window_s` (0,5 s), chỉ báo khi sai số chuẩn của đường hồi quy đủ nhỏ; khoảng
  cách của một track dùng **lớp đồng thuận** (YOLO đổi car↔truck làm khoảng cách nhảy gấp đôi).
- **Phát hiện ảo trên xe mình:** mui + taplo nhận là "car" rộng cả khung hình (AEB "xe 2 m"); vật trang trí trên taplo
  chạm mép dưới nhận là "người 4 m" (cận trên theo mặt đường chỉ đúng khi mép dưới khung là mặt đường). Xe vượt ở khúc
  cua không còn bị coi là cắt làn (`perception.cut_in_max_pull_away_mps`).
- **AEB cần xác nhận** `safety.aeb_confirm_s: 0.1`, tính xuyên qua các lần mất phát hiện ngắn (`aeb_confirm_gap_s: 0.1`: xe cắt làn
  ở 5 m thường bị YOLO bỏ sót vài frame); FCW vẫn tác động ngay.
- **Đo bằng `scripts/gate_replay.py`** (chạy detector 1 lần trên GPU, phát lại hình học + TTC + gate trên CPU): 111 video
  Nexar tập test (mốc cảnh báo/va chạm do người gán) + 24 clip Australian có nhãn + 1 clip cao tốc. Trên các clip lái
  bình thường, AEB sai giảm từ **15,8% thời gian / 8,6 lần mỗi phút** (code cloud) xuống **4,0% / 2,3 lần**; mẫu nhãn
  KEEP bị AEB 16% → 1,5%; SUV cắt làn ở 5 m (clip va chạm) vẫn được AEB lúc 9,3 s. Đánh đổi: trong đoạn nguy hiểm
  của Nexar AEB bật ở 53% video (cửa sổ đối chứng cùng độ dài: 25%) so với 76% (35%) trước đây; FCW hoặc AEB vẫn phản
  ứng ở 90%. `safety.aeb_confirm_s: 0` = không xác nhận.
- **Policy `cautious_gated` không dùng:** trên val v3, under-braking chỉ giảm 8,8% → 7,0% trong khi phanh thừa tăng
  14,8% → 24,2%; hầu hết ca phanh thiếu không có dấu hiệu nguy hiểm nào trong perception.
- **Dataset `ds_v3`** = đúng các mẫu của `ds_v2` build lại với perception mới (`build-dataset comma2k19 --only-from`,
  `scripts/v4_data.sh`), thêm frame 0,5 s trước và `lead`; nhãn/split giữ qua overlay. Model 2 frame v4:
  `scripts/pipeline_v4.sh`.

## Cập nhật 10/10/2026 (phiên cloud, **chưa chạy trên GPU**): Depth-Anything thay khoảng cách pinhole

Khoảng cách từ chiều cao bbox là nguồn nhiễu lớn nhất của perception (vài % mỗi frame, gấp đôi khi YOLO đổi
car↔truck, ước lượng xa hơn thật với bbox bị cắt). `perception/depth.py` chạy một mạng depth dày (Depth-Anything V2,
đúng model Qualcomm AI Hub phát hành dưới tên `depth_anything_v2`) và lấy giá trị bên trong từng bbox:

- Model gốc chỉ cho **inverse depth tương đối** (`1/d = scale·value + shift`). Scale/shift được khôi phục mỗi frame bằng
  hồi quy robust trên các **anchor có khoảng cách đã biết**: điểm mặt đường phía trước xe (hình học mặt phẳng + chiều
  cao camera, cùng công thức đã dùng cho bbox bị cắt) và bbox xe nguyên vẹn với khoảng cách pinhole. Fit được làm mượt
  theo thời gian; frame không fit được giữ khoảng cách pinhole. Checkpoint metric (`...-Metric-Outdoor-...`) bỏ qua
  bước này (`output_kind: metric_depth`).
- Hợp nhất trong `MotionEstimator`: `perception.depth.mode: replace` (bbox bị cắt vẫn không bao giờ xa hơn cận trên
  pinhole) hoặc `min` (không bao giờ xa hơn pinhole → phanh không muộn hơn trước). Nguồn khoảng cách ghi ở
  `Detection.distance_source`, HUD thêm chữ `D` sau khoảng cách.
- Backend `transformers` (checkpoint HF, `python scripts/fetch_hf.py depth-anything/Depth-Anything-V2-Small-hf`,
  Apache-2.0; Base/Large là CC-BY-NC-4.0) hoặc `onnx` (file shape tĩnh, ví dụ export từ AI Hub). Mọi phần sau mạng là
  numpy thuần nên `scripts/gate_replay.py` phát lại được trên CPU.
- Đo false AEB không cần chạy lại detector: `python scripts/gate_replay.py depth outputs/gate_replay` bổ sung thống kê
  depth vào capture sẵn có (đọc lại video), rồi
  `python scripts/gate_replay.py replay outputs/gate_replay --variant pinhole:perception.depth.mode=off --variant depth:perception.depth.mode=replace --variant depth_min:perception.depth.mode=min`.
  Chỉ bật mặc định (`perception.depth.enabled: true`) nếu AEB sai trên clip bình thường giảm mà AEB trong cửa sổ nguy
  hiểm Nexar không giảm.

## Vòng 5 (chuẩn bị, chưa chạy): base model Qwen2.5-VL-7B-Instruct

Qwen2.5-VL-7B là cỡ Qualcomm AI Hub phát hành bản tối ưu (`qwen2_5_vl_7b_instruct`, Genie w4a16, có SA8650P / SA8775P)
và dùng license Apache-2.0 (bản 3B là Qwen Research, phi thương mại). Cùng kiến trúc với 3B nên dataset, prompt,
`encode_messages`, regex LoRA, input 2 frame dùng lại nguyên; chỉ fine-tune lại.

- Profile `configs/vlm_7b.yaml`; toàn bộ vòng chạy bằng `scripts/pipeline_v5_7b.sh` (smoke 20 step đo VRAM + ETA →
  train 3 epoch theo recipe v3 → merge → eval v5 vs v3 trên ds_v3 → sweep → demo → báo cáo HTML).
- `adas-vla train --optimizer paged_adamw_8bit --max-steps N`: optimizer 8-bit của bitsandbytes cho 7B trên 8 GB;
  chạy N step rồi dừng, in VRAM đỉnh và ETA của cả vòng.
- `adas-vla merge` giờ gộp LoRA **theo từng shard** safetensors (`W + α/r·B·A`), không cần nạp cả model 7B bf16
  (16 GB RAM) vào bộ nhớ; `--full` là đường cũ qua peft.
- Nếu smoke hết VRAM: `V5_EXTRA="--set vlm.quantize_vision=true"` cho mọi bước 7B và
  `TRAIN_EXTRA="--optimizer paged_adamw_8bit --lora-r 8"`. v5 chỉ thành mặc định nếu thắng v3 ở under-braking trước.

## Giới hạn và lưu ý

- **Không dùng để điều khiển xe thật.** Đây là prototype R&D.
- Khoảng cách được ước lượng từ 1 camera (chiều cao bbox + FOV), nên nhiễu. Cần chỉnh `camera.hfov_deg` theo camera thật.
  `perception.depth` (Depth-Anything) là lựa chọn thay thế, chưa được đo trên dữ liệu thật.
- Với video, tốc độ ego lấy từ `--ego-speed` (không có CAN bus).
- **License:** Qwen2.5-VL-3B dùng Qwen Research License (phi thương mại); vòng 5 chuyển sang Qwen2.5-VL-7B (Apache-2.0). Ultralytics YOLO dùng AGPL-3.0. Trước khi thương mại hóa, xem mục 3–4 của tài liệu deploy.
