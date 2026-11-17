# Tóm tắt EDA

- Tập train: **3,823** mẫu, 64 đặc trưng pixel và 1 nhãn.
- Tập test chính thức: **1,797** mẫu.
- Ảnh được biểu diễn thành ma trận 8×8; pixel nguyên trong khoảng 0–16.
- Nhãn là chữ số 0–9. Dữ liệu CSV không có header.
- Giá trị thiếu: 0; mọi pixel và nhãn đều được kiểm tra miền hợp lệ khi nạp.
- Tổng số bản ghi trùng hoàn toàn trong train: 0.

## Số mẫu theo lớp

| Chữ số | Train | Test |
|---:|---:|---:|
| 0 | 376 | 178 |
| 1 | 389 | 182 |
| 2 | 380 | 177 |
| 3 | 389 | 183 |
| 4 | 387 | 181 |
| 5 | 376 | 182 |
| 6 | 377 | 181 |
| 7 | 387 | 179 |
| 8 | 380 | 174 |
| 9 | 382 | 180 |
