# Fetal Brain Ultrasound Fuzzy Integral Ensemble

<div align="center">

**Dự án phân loại ảnh siêu âm não thai nhi 16 lớp bằng nhiều mô hình CNN pretrained và ensemble với Fuzzy Integral.**

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-Xception-111827?style=for-the-badge)
![torchvision](https://img.shields.io/badge/torchvision-Models-111827?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 1. Project Title & Catchphrase

**Fetal Brain Ultrasound Fuzzy Integral Ensemble** là dự án phân loại ảnh siêu âm não thai nhi theo 16 lớp bất thường hoặc bình thường.

Dự án huấn luyện nhiều mô hình pretrained bằng PyTorch, sau đó kết hợp xác suất dự đoán bằng các phương pháp ensemble như **Soft Voting**, **Weighted Average**, **Sugeno Fuzzy Integral** và **Choquet Fuzzy Integral**.

> Dự án phục vụ mục đích học tập, nghiên cứu và thử nghiệm mô hình AI. Kết quả dự đoán không thay thế chẩn đoán y khoa.

---

## 2. Quick Demo & Visuals

README gốc chưa cung cấp ảnh demo hoặc ảnh kết quả, nên phần hình ảnh minh họa được bỏ qua.

Các kết quả chính được lưu trong:

```text
outputs/weights/
outputs/logs/
outputs/cache/
```

---

## 3. Tính Năng Nổi Bật

- **Phân loại ảnh siêu âm não thai nhi 16 lớp:** xử lý bài toán multiclass classification trên dữ liệu Roboflow.
- **Huấn luyện 4 base models:** gồm Xception, ResNet-50, DenseNet-121 và VGG16.
- **Transfer learning:** sử dụng pretrained ImageNet và fine-tune trên dữ liệu siêu âm.
- **Ensemble nâng cao:** hỗ trợ Best Single, Soft Voting, Weighted Average, Sugeno Integral và Choquet Integral.
- **Đánh giá chống data leakage:** mọi tham số ensemble được fit trên VALID, TEST chỉ dùng để báo cáo cuối cùng.

---

## 4. Công Nghệ Sử Dụng

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Model%20Training-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Training-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![timm](https://img.shields.io/badge/timm-Xception-111827?style=for-the-badge)
![torchvision](https://img.shields.io/badge/torchvision-ResNet%20DenseNet%20VGG-111827?style=for-the-badge)
![Roboflow](https://img.shields.io/badge/Roboflow-Multiclass%20Dataset-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

### Thành phần kỹ thuật

| Nhóm | Công nghệ | Vai trò |
|---|---|---|
| Base models | Xception, ResNet-50, DenseNet-121, VGG16 | Trích xuất đặc trưng và phân loại ảnh |
| Framework | PyTorch | Huấn luyện và inference mô hình |
| Model libraries | timm, torchvision | Tải backbone pretrained ImageNet |
| Loss | CrossEntropyLoss | Tối ưu bài toán phân loại đa lớp |
| Ensemble | Soft Voting, Weighted Average, Sugeno, Choquet | Kết hợp xác suất dự đoán từ nhiều mô hình |
| Validation strategy | Fit ensemble trên VALID | Tránh tuning trực tiếp trên TEST |
| Notebook | VSCode + Jupyter | Chạy pipeline huấn luyện và đánh giá |

---

## 5. Triển Khai Nhanh

**Prerequisites**

- Windows
- Python 3.11.x
- VSCode và Jupyter extension
- GPU NVIDIA nếu muốn huấn luyện nhanh hơn
- Dataset đã giải nén đúng cấu trúc `train/valid/test` và có `_classes.csv`

```bash
# Di chuyển vào thư mục dự án
cd "D:\Documents\Fuzzy_Integral\fetal_brain_ultrasound_fi_ensemble_16cls"

# Tạo và kích hoạt môi trường ảo trên Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Cập nhật pip
python -m pip install -U pip

# Cài thư viện phụ thuộc
pip install -r requirements.txt

# Nếu gặp xung đột NumPy/OpenCV, cài lại bản ổn định
pip uninstall -y numpy opencv-python
pip install numpy==1.26.4 opencv-python==4.10.0.84

# Mở notebook bằng VSCode hoặc Jupyter
# Chạy lần lượt các cell huấn luyện, đánh giá và ensemble
```

---

## 6. Tài Liệu Dự Án

### Bài toán

| Thành phần | Mô tả |
|---|---|
| Input | Ảnh siêu âm não thai nhi |
| Output | Một trong 16 lớp bất thường hoặc bình thường |
| Task | Multiclass image classification |
| Dataset | Fetal Brain Abnormalities Ultrasound |
| Nguồn | Roboflow |
| Framework | PyTorch |
| Ensemble | Fuzzy Integral và voting-based ensemble |

### Dataset

| Hạng mục | Thông tin |
|---|---|
| Tên dataset | Fetal Brain Abnormalities Ultrasound |
| Nguồn | Roboflow |
| Kiểu export | Multiclass |
| Dữ liệu | Ảnh `.jpg` và file `_classes.csv` cho từng split |
| Quy mô | Khoảng 1.800 ảnh train/valid/test |
| Số lớp | 16 |
| Split | Train / Valid / Test |

> README gốc chưa cung cấp link Roboflow cụ thể. Hãy bổ sung link dataset nếu repository cần người khác tải lại dữ liệu.

### Cấu trúc dữ liệu

```text
data/
└── Fetal Brain Abnormalities Ultrasound.v1i.multiclass/
    ├── train/
    │   ├── *.jpg
    │   └── _classes.csv
    ├── valid/
    │   ├── *.jpg
    │   └── _classes.csv
    ├── test/
    │   ├── *.jpg
    │   └── _classes.csv
    ├── README.dataset.txt
    └── README.roboflow.txt
```

Ghi chú xử lý dữ liệu:

- `_classes.csv` được đọc theo từng split.
- Tên class trong `_classes.csv` có thể có khoảng trắng đầu dòng.
- Pipeline đã chuẩn hóa tên class khi đọc để tránh lỗi sai nhãn.

### Base models

| Model | Library | Ghi chú |
|---|---|---|
| Xception | timm | Dùng biến thể `legacy_xception` |
| ResNet-50 | torchvision.models | Pretrained ImageNet |
| DenseNet-121 | torchvision.models | Pretrained ImageNet |
| VGG16 | torchvision.models | Pretrained ImageNet |

### Training pipeline

| Thành phần | Cấu hình |
|---|---|
| Split | Train / Valid / Test |
| Loss | CrossEntropyLoss |
| Class imbalance | Có thể dùng class weights nếu lệch lớp |
| Optimizer | Theo config notebook |
| Scheduler | Theo config notebook |
| Model selection | Lưu best weights theo metric trên VALID |
| Metric ưu tiên | Macro-F1 |

### Ensemble methods

| Phương pháp | Mô tả |
|---|---|
| Best single | Chọn mô hình đơn tốt nhất theo VALID |
| Soft voting | Trung bình xác suất dự đoán |
| Weighted average | Gán trọng số toàn cục theo VALID metric |
| Sugeno Fuzzy Integral | Fusion theo min-max và fuzzy measure |
| Choquet Fuzzy Integral | Fusion theo fuzzy measure có xét tương tác nguồn |
| Choquet Init | Khởi tạo fuzzy density `g` từ VALID metric rồi giải lambda |
| Choquet OPT | Random search tối ưu `g` trên VALID, ràng buộc tổng `g = 0.9`, sau đó giải lambda |

### Nguyên tắc đánh giá công bằng

| Quy tắc | Mục tiêu |
|---|---|
| Fit ensemble weights trên VALID | Tránh dùng TEST để chọn tham số |
| Fit fuzzy density `g` trên VALID | Tránh data leakage |
| Fit lambda trên VALID hoặc từ `g` đã chọn | Giữ TEST độc lập |
| TEST chỉ dùng báo cáo cuối | Đánh giá khách quan hơn |

### Kết quả đầu ra

| Thành phần | Đường dẫn |
|---|---|
| Best weights từng model | `outputs/weights/` |
| Training logs | `outputs/logs/` |
| Probability hoặc prediction cache | `outputs/cache/` |
| Bảng so sánh ensemble | Trong notebook hoặc file export nếu có |
| Báo cáo metric | Trong notebook hoặc file export nếu có |

### Cấu trúc dự án gợi ý

```text
fetal_brain_ultrasound_fi_ensemble_16cls/
├── data/
│   └── Fetal Brain Abnormalities Ultrasound.v1i.multiclass/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── README.dataset.txt
│       └── README.roboflow.txt
├── outputs/
│   ├── weights/
│   ├── logs/
│   └── cache/
├── notebooks/
├── requirements.txt
└── README.md
```

> Cấu trúc trên được chuẩn hóa từ README gốc. Nếu repository thực tế có thêm thư mục `src/`, `configs/` hoặc `reports/`, hãy cập nhật lại cho khớp.

### Git ignore khuyến nghị

```text
.venv/
data/
outputs/weights/
outputs/cache/
*.pth
*.pt
*.pkl
__pycache__/
.ipynb_checkpoints/
```

### Lưu ý sử dụng

- Không commit dataset, model weights hoặc file cache nặng lên GitHub nếu repository public.
- Kiểm tra lại tên thư mục dataset nếu khác `Fetal Brain Abnormalities Ultrasound.v1i.multiclass`.
- Kiểm tra lại các đường dẫn trong notebook trước khi chạy trên máy khác.
- Với bài toán y tế, cần kiểm tra data leakage, trùng ảnh và tính độc lập giữa train/valid/test trước khi dùng kết quả cho báo cáo.
- Dự án chỉ phục vụ học tập và nghiên cứu, không dùng để thay thế chẩn đoán y khoa.

### Bản quyền

Copyright by **ANH PHAP TO (Franceto)**.

### Tác giả

**Franceto (ANH PHAP TO)**

### Support

Nếu project hữu ích, hãy cho repository một sao.

Made by **Franceto (ANH PHAP TO)**
