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
  perception/         detector.py (YOLO11s+ByteTrack, màu đèn), geometry.py (distance/TTC/oncoming), lanes.py (YOLOP CNN hoặc classic)
  reasoning/          prompts.py, parser.py (JSON chịu lỗi), vlm.py (transformers, 4-bit, LoRA)
  control/            safety.py (safety gate), controller.py
  pipeline.py         dual-rate pipeline, VLM sync/async
  hud.py  sources.py  cli.py
  training/           autolabel.py, finetune.py (QLoRA + cân bằng lớp), evaluate.py, data.py
  datasets/           comma2k19.py (nhãn từ CAN), clips.py (teacher + safety), common.py
  review.py report.py trang review nhãn · báo cáo HTML so sánh các lần đánh giá
  deploy/export.py    ONNX + calibration cho QAIRT
  reasoning/llm.py    LLM text-only giải thích sự kiện ADAS · events.py: tách sự kiện từ log
configs/              default.yaml (Qwen2.5-VL-3B + Qwen3-4B, 4-bit) · smoke.yaml · pc_fast.yaml · teacher_7b.yaml
scripts/              download_samples.sh · download_models.sh · fetch_hf.py
docs/DEPLOY_SA8797P.md
```

## Kết quả đã kiểm chứng trên PC này (RTX 4060 Laptop 8 GB, 27/09/2026)

| Thành phần | Kết quả |
|---|---|
| Unit test | 30/30 pass (`pytest -q`) |
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

## Giới hạn và lưu ý

- **Không dùng để điều khiển xe thật.** Đây là prototype R&D.
- Khoảng cách được ước lượng từ 1 camera (chiều cao bbox + FOV), nên nhiễu. Cần chỉnh `camera.hfov_deg` theo camera thật.
- Với video, tốc độ ego lấy từ `--ego-speed` (không có CAN bus).
- **License:** Qwen2.5-VL-3B dùng Qwen Research License (phi thương mại). Ultralytics YOLO dùng AGPL-3.0. Trước khi thương mại hóa, xem mục 3–4 của tài liệu deploy.
