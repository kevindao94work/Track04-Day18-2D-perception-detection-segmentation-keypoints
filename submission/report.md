# Lab 18 Report

## Link notebook đã chạy

Link notebook đã chạy:
https://github.com/kevindao94work/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

## Core checks và môi trường

Tất cả 11 checks gồm các hàm tự cài đặt, FLIP_IDX, auto-label và bonus AP đều ở trạng thái `ok`, không dùng lifeline. Các gate 1B, 2B, 3B, 3C, 4A đã pass; Q1–Q12 đã trả lời bằng tiếng Việt dựa trên các output thực tế.

Môi trường Colab: GPU Tesla T4, torch 2.11.0+cu130, torchvision 0.26.0+cu130, ultralytics 8.4.171. Dùng Google Colab CLI với một runtime persistent; tất cả inference/training/checks/benchmark chạy remote. Notebook được chạy theo từng section và dependent cells được chạy lại khi chỉnh sửa; không thực hiện thêm một lần Restart-and-Run-All gây retraining 40 epochs.

### Latency detector — T4, trung bình 30 lần

| Cấu hình | preprocess (ms) | inference (ms) | postprocess (ms) | số box |
| --- | --- | --- | --- | --- |
| one-to-many + NMS, conf 0.25 | 1.750000 | 9.100000 | 1.140000 | 5 |
| one-to-many + NMS, conf 0.001 | 1.720000 | 8.820000 | 1.270000 | 203 |
| one-to-one NMS-free, conf 0.25 | 1.750000 | 9.210000 | 0.410000 | 5 |
| one-to-one NMS-free, conf 0.001 | 1.680000 | 8.800000 | 0.390000 | 204 |

### Fine-tune tiger-pose — 40 epochs / 640 / seed 0

| flip_idx | epochs | imgsz | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 | train (phút) |
| --- | --- | --- | --- | --- | --- | --- |
| giải phẫu | 40 | 640 | 0.930349 | 0.995000 | 0.457346 | 4.400000 |

Train 210 ảnh, val 53 ảnh; cả hai tập đều chỉ có hổ quay phải. OKS trung bình 0.745000, missed 0/53, suspected left/right swaps 4/53 (chỉ là heuristic khi swapped OKS tăng hơn 0,02). Sáu ảnh tệ nhất và biểu đồ sai số được giữ trong notebook; Q11 phân tích Frame_54.jpg và Frame_31.jpg. Các điểm paw khó hơn các điểm trên trục giữa; mAP50 cao không có nghĩa mAP50–95 cũng cao.

## Bonus 1D

`average_precision` tự cài đặt pass 4 phép thử và giữ COCO 101-point interpolation; AP ví dụ slide = 0.535007; PR curve được lưu trong notebook. Không dùng implementation thư viện để thay student function.

## Bonus 4C

Hai model đều train 40 epochs, imgsz 640, seed 0: flip_idx giải phẫu và flip_idx đồng nhất. Chỉ khác quy ước flip_idx; dùng cùng train/val và đánh giá thêm mirrored val với nhãn đổi theo quy ước giải phẫu.

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương |
| --- | --- | --- |
| flip_idx giải phẫu | 0.457346 | 0.438848 |
| flip_idx đồng nhất | 0.416922 | 0.297723 |

Trong run này, mapping giải phẫu giảm 0.018498 khi lật gương, trong khi mapping đồng nhất giảm 0.119199; mirrored val bộc lộ lỗi rõ hơn dù val gốc vẫn đạt điểm đáng kể.

Các cột là **Pose mAP50–95**, không phải Pose mAP50. Val gốc có cùng bias orientation như train nên có thể che lỗi mapping trái/phải; mirrored val kiểm tra tính nhất quán ngữ nghĩa khi đổi hướng. Kết quả trên một distribution hẹp không đảm bảo generalization sang viewpoint hoặc tư thế mới; không chọn seed hay sửa metric để khớp reference.

## kpt_oks_sigmas exploration

Giữ model đã train, tạo vector proxy từ MAD của residual (x,y) dự đoán đã chuẩn hóa theo √(0,53 × box area), chia cho 2 để khớp k=2σ, giới hạn [0,01; 0,3], rồi thêm `kpt_oks_sigmas` vào YAML riêng và chạy validation lại.

| keypoint | equal_sigma | proxy_sigma |
| --- | --- | --- |
| nose | 0.083333 | 0.014861 |
| head | 0.083333 | 0.013173 |
| withers | 0.083333 | 0.013260 |
| tail_base | 0.083333 | 0.013070 |
| right_hind_hock | 0.083333 | 0.048801 |
| right_hind_paw | 0.083333 | 0.073386 |
| left_hind_paw | 0.083333 | 0.113692 |
| left_hind_hock | 0.083333 | 0.074833 |
| right_front_wrist | 0.083333 | 0.025172 |
| right_front_paw | 0.083333 | 0.132842 |
| left_front_wrist | 0.083333 | 0.030754 |
| left_front_paw | 0.083333 | 0.088969 |

| condition | Pose_mAP50 | Pose_mAP50_95 |
| --- | --- | --- |
| equal 1/12 | 0.995000 | 0.457346 |
| empirical residual proxy | 0.249870 | 0.051989 |

**Giới hạn:** true OKS sigmas nên được ước lượng từ inter-annotator variability, không phải model residual. Dataset không có repeated independent annotations. Proxy này trộn bias/lỗi localization của model với độ mơ hồ nhãn; cùng val được dùng để ước lượng và đánh giá nên đây là exploration, không phải calibration độc lập và không chứng minh model tốt lên.

## Homework — Auto-label + YOLO26n-seg

Nguồn: [Ultralytics COCO128](https://docs.ultralytics.com/datasets/detect/coco128/), một subset nhỏ của MS COCO 2017; [COCO terms of use](https://cocodataset.org/#termsofuse). Giữ manifest COCO image IDs để tái tạo subset, không commit ảnh gốc. Ảnh nguồn có thể đã nằm trong dữ liệu pretraining COCO, nên experiment này minh họa pipeline teacher/student, không phải đánh giá độc lập trên domain mới.

Chọn 25 ảnh đầu tiên theo filename có person được YOLO26n phát hiện ở conf 0,5 và có label vượt QA. Pipeline detector → box prompt SAM 2.1 → student mask_to_yolo_seg → polygon_to_mask/mask_iou round-trip. QA kiểm tra empty/tiny/huge mask, polygon <3 đỉnh, tọa độ không hợp lệ, duplicate và round-trip IoU; giảm simplification để sửa polygon khi hữu ích, các trường hợp chưa sửa được khiến toàn bộ ảnh bị loại và được ghi trong QA log để tránh nhãn thiếu người. Visual QA kiểm tra 9 ảnh bị flag/correction; 5 label được sửa polygon (8 lần thử gồm cả label sau đó bị loại), và 2 trong số này được sửa thêm bằng positive/negative prompts: khôi phục bàn tay ở 000000000326.jpg, loại broccoli khỏi mask person ở 000000000370.jpg. Hai mask sửa prompt có round-trip IoU 0,994651 và 0,985798; hình trước/sau và QA contact sheet được giữ trong notebook/artifacts. Split bằng RNG seed 0, không trùng image ID: 20 train / 5 val.

| Thông số | Giá trị |
| --- | --- |
| source | Ultralytics coco128 subset of MS COCO 2017 |
| source_url | https://cocodataset.org/#termsofuse |
| images | 25 |
| target_class | person |
| labels | 64 |
| polygon_corrections | 5 |
| polygon_correction_attempts | 8 |
| manual_prompt_corrections | 2 |
| flagged_instances | 4 |
| train_images | 20 |
| val_images | 5 |
| epochs | 15 |
| imgsz | 640 |
| mask_map50 | 0.370147 |
| mask_map50_95 | 0.219696 |
| limitation | Validation uses SAM pseudo-labels, not independently corrected ground truth; small sample and detector selection bias. |

Mask mAP này đo so với **SAM pseudo-labels**, không phải ground-truth accuracy. Số ảnh nhỏ, selection bias của detector và lỗi teacher hạn chế kết luận. Manifest, log correction/rejection và metrics nằm ở `submission/bonus/segmentation.json`; labels được lưu ở `submission/bonus/person_labels.zip`.

## Homework — ONNX latency

Export YOLO26n thành hai graph riêng bằng Ultralytics 8.4.171: nms=None cho raw one-to-many, nms=False cho end-to-end one-to-one. Kiểm tra output shapes thật: {'one-to-many': [1, 84, 8400], 'one-to-one': [1, 300, 6]}. Không dùng GPU provider: CPUExecutionProvider, ONNX Runtime 1.30.0; CPU Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical CPUs. intra-op 2 threads, inter-op 1 thread, input 1×3×640×640 float32, 10 warm-up / 30 measured iterations mỗi cấu hình.

| head | confidence | preprocess_ms | inference_ms | postprocess_ms | total_ms | boxes | iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one-to-many | 0.250000 | 4.932077 | 76.806810 | 5.396302 | 87.135189 | 5 | 30 |
| one-to-many | 0.001000 | 5.526225 | 114.987221 | 75.678208 | 196.191654 | 186 | 30 |
| one-to-one | 0.250000 | 3.913957 | 65.403608 | 0.510397 | 69.827962 | 5 | 30 |
| one-to-one | 0.001000 | 3.940873 | 68.600466 | 0.532922 | 73.074260 | 177 | 30 |

Full pipeline gồm letterbox/RGB/normalize → ONNX forward → postprocess/coordinate scaling. One-to-many dùng student class-aware greedy NMS với IoU 0,7 và max 300 detections; one-to-one chỉ lọc score và scale tọa độ. Đây là latency của implementation NMS tự viết, không đại diện cho mọi NMS tối ưu; input square khác benchmark core dùng rectangular letterbox nên không so trực tiếp total giữa hai bảng. Đo nhiều lần nhưng CPU shared runtime và tải hệ thống vẫn gây nhiễu; không bao gồm đọc file ảnh hoặc export graph.

## Artifacts và giới hạn xác minh

Notebook giữ outputs checks, plots, tables, training/error analysis và homework. `submission/ket_qua.json` do final_report() tạo, không manual-edit. Checkpoints best/last và results được backup trong Google Drive `Lab18_Backup`; không commit weights hoặc credentials. Đã chạy cả hai homework; rubric chỉ cộng tối đa một homework 5 điểm, tổng điểm tối đa vẫn là 120, điểm cuối cùng do người chấm xác định.
