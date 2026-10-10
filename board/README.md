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

`$DIR` trên board phải có `lib/` (libQnnHtp.so, libQnnHtpV73Stub.so, libQnnSystem.so cho aarch64-qnx800) và
`models/` (`yolo11s_w8a8.bin`, `yolop_lane_w8a16_o3v8.bin`); đổi bằng biến `DIR`, `DET`, `LANE`, `PORT` (mặc định
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

## Giao thức

Xem đầu file `vla_stream_server.cpp` ("VLASTREAM 1"). `third_party/stb_image.h` = stb_image v2.30 (public domain /
MIT, nothings/stb), sha256 `594c2fe3…00b3`.
