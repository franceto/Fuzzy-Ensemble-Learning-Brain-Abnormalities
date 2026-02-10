# Copyright by ANH PHAP TO (Franceto)

# Fetal Brain Ultrasound – Fuzzy Integral Ensemble (PyTorch)

## 1) Giới thiệu bộ dữ liệu

- **Tên dataset (Roboflow):** _Fetal Brain Abnormalities Ultrasound_ (multiclass)
- **Bài toán:** Phân loại ảnh siêu âm thai nhi theo **16 lớp** bất thường/normal.
- **Quy mô:** ~1.8k ảnh (train/valid/test).
- **Dạng export đang dùng:** `multiclass` gồm ảnh `.jpg` + file `_classes.csv` cho từng split.
- **Cấu trúc dữ liệu (sau khi giải nén):**
  data/
  Fetal Brain Abnormalities Ultrasound.v1i.multiclass/
  train/ (jpg + \_classes.csv)
  valid/ (jpg + \_classes.csv)
  test/ (jpg + \_classes.csv)
  README.dataset.txt
  README.roboflow.txt
- **Ghi chú quan trọng:** Tên class trong `_classes.csv` đôi khi có khoảng trắng đầu dòng → pipeline đã chuẩn hoá khi đọc.

---

## 2) Mô hình / kỹ thuật / thuật toán sử dụng

### 2.1 Base models (transfer learning)

Huấn luyện 4 mô hình phân loại ảnh bằng **PyTorch** (pretrained ImageNet, fine-tune):

- **Xception** (qua `timm`, dùng biến thể legacy_xception)
- **ResNet-50** (`torchvision.models`)
- **DenseNet-121** (`torchvision.models`)
- **VGG16** (`torchvision.models`)

### 2.2 Training pipeline (chuẩn nghiên cứu)

- Split rõ ràng: **train / valid / test**
- Loss: CrossEntropyLoss (có thể dùng class weights nếu lệch lớp)
- Optimizer + scheduler (theo config notebook)
- Lưu **best weights** theo metric trên **VALID** (ưu tiên **macro-F1**)

### 2.3 Ensemble methods

Sau khi huấn luyện, trích **probability outputs** cho từng mẫu để fuse:

- **Best single** (mô hình tốt nhất theo VALID)
- **Soft voting** (trung bình xác suất)
- **Weighted average** (trọng số toàn cục theo VALID metric)
- **Sugeno Fuzzy Integral** (fusion theo min–max + fuzzy measure)
- **Choquet Fuzzy Integral**
- **Choquet Init:** khởi tạo fuzzy density `g` từ VALID metric rồi giải **λ**
- **Choquet OPT:** random search tối ưu `g` trên VALID (ràng buộc ∑g = 0.9), giải **λ** cho mỗi ứng viên, khoá tham số và báo cáo trên TEST

### 2.4 Nguyên tắc đánh giá công bằng

- Mọi tham số liên quan ensemble (**weights**, **g**, **λ**, calibration nếu có) đều **fit trên VALID**
- **TEST chỉ dùng báo cáo cuối**, tránh data leakage

---

## 3) Hướng dẫn chạy dự án (VSCode + Jupyter + CUDA)

### 3.1 Yêu cầu

- Windows + GPU NVIDIA (CUDA)
- Python **3.11.x** (khuyến nghị)
- VSCode + Jupyter extension

### 3.2 Tạo dự án & đặt dữ liệu

1. Tạo thư mục dự án, ví dụ:
   D:\Documents\Fuzzy_Integral\fetal_brain_ultrasound_fi_ensemble_16cls

2. Copy dataset đã giải nén vào đúng chỗ:
   D:\Documents\Fuzzy_Integral\fetal_brain_ultrasound_fi_ensemble_16cls\data
   Fetal Brain Abnormalities Ultrasound.v1i.multiclass
   train\ valid\ test\ ...

> Nếu tên folder dataset của bạn khác, chỉ cần đảm bảo cấu trúc train/valid/test + `_classes.csv` đúng như trên.

### 3.3 Tạo môi trường ảo

Mở PowerShell tại thư mục project:
powershell
cd "D:\Documents\Fuzzy_Integral\fetal_brain_ultrasound_fi_ensemble_16cls"
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip

### 3.4 Cài thư viện

pip install -r requirements.txt
Lưu ý: nếu gặp xung đột NumPy/OpenCV, dùng bản ổn định:
pip uninstall -y numpy opencv-python
pip install numpy==1.26.4 opencv-python==4.10.0.84

### 3.5 Kết quả

Weights tốt nhất từng model: outputs/weights/
Logs training: outputs/logs/
Probabilities / predictions cache (nếu dùng): outputs/cache/
Báo cáo metric + bảng so sánh ensemble: trong notebook (hoặc export)
