# Nhận dạng chữ số viết tay

Dự án thực hiện Đề tài 5 trong bài tập lớn Nhập môn Học máy: phân loại chữ số 0–9 từ bộ dữ liệu Optical Recognition of Handwritten Digits.

## Cấu trúc

```text
data/raw/                  Dữ liệu train/test chính thức
src/run_experiments.py     EDA, tiền xử lý, dò tham số, đánh giá và phân tích lỗi
tests/test_pipeline.py     Kiểm tra cấu trúc dữ liệu và đặc trưng
report/main.tex            Báo cáo LaTeX tiếng Việt
outputs/                   Biểu đồ, bảng số liệu và mô hình sinh ra khi chạy
```

## Cài đặt và chạy

Dùng Python 3.10 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_experiments
```

Lệnh chạy đọc hai tệp CSV trong `data/raw`, tạo EDA và biểu đồ trong `outputs/`, chọn tham số trên tập validation tách có phân tầng từ tập train, sau đó đánh giá các cấu hình đã định trước trên test set chính thức. Test set không tham gia chọn tham số. Hạt giống mặc định là `42`.

Để chạy kiểm tra:

```powershell
python -m unittest discover -s tests -v
```

## Thí nghiệm

- **Biểu diễn pixel (baseline):** ma trận 8×8 được làm phẳng và chia 16.
- **Biểu diễn HOG:** HOG 9 hướng, ô 2×2 pixel, block 2×2 ô trên ảnh 8×8.
- **Mô hình:** K-NN (dò `k` và khoảng cách Euclidean/Manhattan) và Decision Tree (dò `max_depth` cùng `min_samples_leaf`).
- **Chọn tham số:** validation 20% tách phân tầng từ tập train; tối ưu macro F1, dùng accuracy để phá hòa.
- **Đánh giá cuối:** accuracy, macro precision/recall/F1, confusion matrix và các mẫu dự đoán sai trên test set chính thức.

Chạy lại chương trình trước khi biên dịch báo cáo để sinh các số liệu và hình được `report/main.tex` tham chiếu. Điền tên thành viên, mã sinh viên và thông tin lớp theo mẫu của học phần trước khi nộp.

Biên dịch báo cáo bằng XeLaTeX (cần cài MiKTeX hoặc TeX Live và font Times New Roman):

```powershell
Set-Location report
xelatex main.tex
xelatex main.tex
```
