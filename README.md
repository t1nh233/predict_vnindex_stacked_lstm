# Dự báo VN-Index sử dụng Stacked LSTM

Dự án này sử dụng mô hình học sâu Stacked LSTM để dự báo chỉ số chứng khoán VN-Index, kết hợp tối ưu hóa siêu tham số bằng thuật toán TPE (Optuna).

## Cấu trúc thư mục
- `data/`: Chứa dữ liệu lịch sử VN-Index.
- `notebooks/`: Chứa file code chạy thực nghiệm chính.
- `src/`: Mã nguồn mô hình (model.py) và hàm tiện ích (utils.py).
- `models/`: Lưu trọng số mô hình (.pth) và bộ scaler (.pkl).
- `results/`: Kết quả biểu đồ và báo cáo.

## Cách chạy
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy file notebook trong thư mục `notebooks/`.
