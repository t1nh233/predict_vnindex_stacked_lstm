# 📈 Dự báo VN-Index bằng Stacked LSTM

Dự án này áp dụng mô hình học sâu **Stacked LSTM** để dự báo chỉ số chứng khoán **VN-Index**, đồng thời sử dụng **Optuna (TPE)** để tối ưu hóa siêu tham số.

---

## 📂 Cấu trúc thư mục

- `data/` : Chứa dữ liệu lịch sử VN-Index (CSV).
- `notebooks/` : Notebook Jupyter/Colab để thực nghiệm và huấn luyện mô hình.
- `src/` : Mã nguồn chính gồm:
  - `model.py` : Định nghĩa kiến trúc LSTM.
  - `utils.py` : Các hàm tiện ích xử lý dữ liệu, huấn luyện, đánh giá.
- `models/` : Lưu trọng số mô hình (`.pth`), cấu hình siêu tham số (`.json`) và scaler (`.pkl`).
- `results/` : Kết quả huấn luyện, biểu đồ, báo cáo.
- `figures/` : Hình ảnh minh họa, biểu đồ loss/accuracy.

---

## 🚀 Cách chạy

1. **Clone repo về máy hoặc Colab**
   ```bash
   git clone https://github.com/t1nh233/predict_vnindex_stacked_lstm.git
   cd predict_vnindex_stacked_lstm
   
2. **Cài đặt thư viện cần thiết**
   ```bash
   pip install -r requirements.txt
   
3. **Chạy dự báo với input**
   ```bash
   python predict.py --input "file_path"
   

    

