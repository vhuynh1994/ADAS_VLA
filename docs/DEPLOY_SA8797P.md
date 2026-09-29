# Deploy ADAS-VLA lên Snapdragon Ride Elite (SA8797P)

> Tài liệu này chỉ dựa trên **nguồn công khai** (liệt kê ở cuối): tài liệu công khai của Qualcomm AI Engine Direct
> (QAIRT) và Qualcomm AI Hub, Ultralytics, ONNX Runtime, AIMET, và model card trên Hugging Face (kiểm tra 09/2026).
> Các thông số cần xác nhận với Qualcomm được liệt kê ở mục 6.

## 1. Tóm tắt

| Hạng mục | Mức sẵn sàng | Ghi chú |
|---|---|---|
| Detector (YOLO), tracker, lane, tính khoảng cách/TTC, safety gate, controller | **Deploy được ngay** | Detector chạy trên HTP (INT8 / W8A16); phần còn lại là C++ thuần trên CPU |
| LLM text-only (giải thích cảnh báo, trợ lý) | **Có đường deploy rõ** | Qualcomm AI Hub có tutorial chạy LLM bằng Genie (`genie-t2t-run`) trên HTP |
| VLM (hiểu cảnh, quyết định) | **R&D** | Cần tách vision encoder + language model, cố định độ phân giải, lượng tử hóa 4-bit |
| VLA end-to-end (quỹ đạo liên tục) | **Nghiên cứu** | Diffusion / flow-matching action decoder chưa có đường deploy công khai |

Nguyên tắc thiết kế giữ nguyên khi lên xe: **VLM chỉ khuyến nghị meta-action; safety gate và controller
deterministic mới là bên quyết định** (có thể kiểm chứng, đạt yêu cầu an toàn chức năng).

## 2. Đường deploy (toolchain công khai)

```
PyTorch / HF model
  └─► (LLM/VLM) lượng tử hóa bằng AIMET, export ONNX + encodings, shape cố định
  └─► (CNN) export ONNX shape cố định + dữ liệu calibration        ← adas-vla export
        └─► qairt-converter → qairt-quantizer → qnn-context-binary-generator
              └─► chạy bằng QNN HTP backend (CNN, vision encoder) hoặc Genie (LLM)
```

- Qualcomm AI Engine Direct hỗ trợ dòng SoC ô tô **SA8295 → SA8797**; với SA8797, tài liệu công khai mô tả môi
  trường **QC Linux (primary VM)** và Android/Linux guest VM. Tài liệu cho **QNX** là add-on riêng.
- Ultralytics export trực tiếp YOLOv8/YOLO11/YOLO26 sang QNN (W8A16) cho các kiến trúc HTP v68 → v81.
- ONNX Runtime có **QNN Execution Provider** nếu muốn giữ runtime ONNX.

## 3. Đánh giá từng module của project

| Module (file) | Chạy ở đâu | Độ khó | Cách làm |
|---|---|---|---|
| Detector `perception/detector.py` | HTP | ★ Dễ | `adas-vla export` → ONNX static `[1,3,384,640]` + calibration → qairt-converter/quantizer → context binary. **License:** Ultralytics là AGPL-3.0 |
| Lane `perception/lanes.py` (YOLOP) | HTP | ★ Dễ | ONNX có sẵn (MIT); chỉ giữ 2 head segmentation |
| Tracker (ByteTrack) | CPU | ★ Dễ | Port C++ |
| Geometry / TTC `perception/geometry.py` | CPU | ★ Dễ | Port C++; nên thay khoảng cách mono bằng depth/radar fusion |
| Safety gate `control/safety.py` | CPU / safety island | ★ Dễ (logic), cần đạt ASIL | Viết lại C++ deterministic, giữ bộ test tương đương `tests/test_safety_control.py` |
| Controller `control/controller.py` | CPU / MCU | ★ Dễ | Thay P-controller bằng PID/MPC đã hiệu chỉnh |
| VLM `reasoning/vlm.py` | HTP (+ CPU ghép embedding) | ★★★ Khó | Tách vision encoder và LM, lượng tử hóa W4A16, KV cache, cố định 560×308 |
| LLM `reasoning/llm.py` | HTP (Genie) | ★★ Trung bình | Theo tutorial LLM-on-Genie của AI Hub |
| Fine-tune `training/` | Không chạy trên xe | — | Train trên PC/server, merge LoRA trước khi export |

## 4. Chọn model VLM / LLM

Tiêu chí: vừa bộ nhớ khi lượng tử hóa, dễ tách thành các sub-model, license phù hợp mục đích sử dụng.

| Model | License (HF) | Params | Cách ghép ảnh vào LLM | Nhận xét |
|---|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** (mặc định) | Qwen Research (phi thương mại) | 3.75B | Chỉ ở **embedding đầu vào** | Tách gọn thành ViT + LM; W4 LM ≈ 2 GB. Phù hợp R&D/học tập |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | 8.3B | Như trên | Dùng thương mại được; nặng hơn (W4 ≈ 4 GB) |
| Qwen3-VL-2B / 4B-Instruct | Apache-2.0 | 2.1B / 4.4B | **DeepStack**: feature ảnh tiêm vào nhiều layer đầu của LLM | Tách sub-model phức tạp hơn; cần xác nhận hỗ trợ |
| Qwen3.5-2B / 4B | Apache-2.0 | 2.3B / 4.7B | Native multimodal, attention lai (linear) | Cần xác nhận hỗ trợ op |
| Qwen-Drive-1.0-4B | Apache-2.0 | 4.5B | Thêm BEV head + planner flow-matching | Tham khảo nghiên cứu |
| **Qwen3-4B-Instruct-2507** (LLM giải thích) | Apache-2.0 | 4.0B | — | Tiếng Việt tốt; W4 ≈ 2.3 GB |

VLA end-to-end công khai (Qwen-Drive, NVIDIA Alpamayo 1.5 / 2) xuất **quỹ đạo liên tục** bằng diffusion hoặc
flow-matching và cần nhiều camera; project này dùng **meta-action rời rạc + safety gate** để deploy được sớm.
Hướng mở rộng: thêm action head nhỏ (MLP) xuất quỹ đạo từ hidden state của LM — một bước, không lặp.

## 5. Checklist khi chuẩn bị deploy VLM

- [ ] Chốt độ phân giải ảnh cố định (`vlm.image_size`, bội số 28 cho Qwen2.5-VL) và fine-tune/đánh giá đúng độ phân giải đó
- [ ] Merge LoRA vào weights (`adas-vla merge`) trước khi lượng tử hóa
- [ ] Rút gọn system prompt, cân nhắc cache prefix cố định
- [ ] Output ngắn: action trước, dừng sớm (`vlm.generate_reason: false`) — đã đo p50 0.9 s trên RTX 4060
- [ ] Dữ liệu calibration lượng tử hóa lấy từ kịch bản ADAS thật (đêm, mưa, đô thị, cao tốc)
- [ ] So sánh lại `adas-vla eval` giữa bản PC và bản lượng tử hóa: accuracy, **under-braking**, latency, bộ nhớ

## 6. Cần xác nhận với Qualcomm

1. Kiến trúc HTP, số core NPU và giới hạn bộ nhớ mỗi process trên SA8797P.
2. Genie / QNN HTP backend có hỗ trợ đầy đủ trên hệ điều hành đích (QNX hay Linux VM) không.
3. Có recipe/model pre-compiled cho Qwen2.5-VL, Qwen3-VL (DeepStack), Qwen3-4B trên Ride Elite không.
4. Qualcomm AI Hub đã có context binary cho chipset Ride Elite chưa.

## Nguồn công khai

- Qualcomm AI Runtime (QAIRT) — Auto platform overview: https://docs.qualcomm.com/doc/80-63442-10/topic/auto_overview.html
- Qualcomm AI Hub models / apps (LLM on Genie): https://github.com/qualcomm/ai-hub-models, https://github.com/quic/ai-hub-apps
- AIMET (lượng tử hóa): https://github.com/quic/aimet
- Ultralytics — YOLO export to QNN: https://docs.ultralytics.com/integrations/qnn
- ONNX Runtime — QNN Execution Provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- Snapdragon Ride Elite: https://www.qualcomm.com/automotive/products/elite
- Model cards: Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3-4B-Instruct-2507, Qwen-Drive-1.0-4B, NVIDIA Alpamayo (Hugging Face)
