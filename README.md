# 🎬 Ứng Dụng Mô Hình Học Sâu Vào Xây Dựng Hệ Thống Gợi Ý Phim (NeuMF)

## 🌟 Giới Thiệu

Hệ thống giải quyết bài toán gợi ý **Top-K Recommendation** với dữ liệu tương tác ẩn (**Implicit Feedback**). Dự án sử dụng mô hình **NeuMF**, tích hợp hai nhánh học bổ trợ cho nhau:

1. 
**Generalized Matrix Factorization (GMF):** Nắm bắt các tương tác tuyến tính thông qua phép nhân từng phần tử của các vector nhúng (Embeddings).


2. 
**Multi-Layer Perceptron (MLP):** Học các tương tác phi tuyến tính phức tạp bằng mạng nơ-ron nhiều lớp.



Hệ thống được phát triển hoàn chỉnh **End-to-End** từ tiền xử lý dữ liệu, huấn luyện mô hình cho đến triển khai giao diện Web thông qua Flask Backend.

---

## 📁 Cấu Trúc Thư Mục Triển Khai

Để hệ thống hoạt động chính xác, cấu trúc cây thư mục dự án cần được sắp xếp như sau:

```text
📂 project/
├── 📂 data/
│   ├── 📄 movies.csv           # Dữ liệu thông tin phim (movieId, title, genres)
│   ├── 📄 ratings.csv          # Dữ liệu tương tác người dùng (userId, movieId, rating)
├── 📂 output/
│   ├── 📄 neumf_final.pth      # Trọng số mô hình NeuMF lưu sau khi train tốt nhất
│   ├── 📄 user_mapping.pkl     # Tệp ánh xạ nhãn User ID sang chỉ mục Embedding
│   └── 📄 item_mapping.pkl     # Tệp ánh xạ nhãn Movie ID sang chỉ mục Embedding
├── 📂 templates/
│   └── 📄 index.html           # Giao diện Frontend giao tiếp trực tiếp với Flask UI
├── 📄 config.py                # Cấu hình siêu tham số (EMB_DIM, MLP_LAYERS, DEVICE)
├── 📄 train.py                 # Mã nguồn tiền xử lý và huấn luyện mô hình
└── 📄 app.py                   # Flask Backend API và Web Demo

```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Chuẩn bị môi trường

Yêu cầu cài đặt **Python** và các thư viện cần thiết. Bạn có thể cài đặt nhanh qua pip:

```bash
pip install torch pandas numpy flask pickle-mixin

```

### 2. Tiền xử lý & Huấn luyện mô hình( Đã sẵn có vậy nên không cần nữa)

Chạy kịch bản huấn luyện để xử lý dữ liệu, chia tập theo phương pháp `Leave-One-Out`, áp dụng `Negative Sampling` và huấn luyện mô hình NeuMF:

```bash
python train.py

```

Sau khi hoàn tất, các tệp trọng số `neumf_final.pth` và ánh xạ `user_mapping.pkl`, `item_mapping.pkl` sẽ được lưu tự động vào thư mục `/output/`.

### 3. Kích hoạt Web Demo

Khởi chạy Flask server để chạy giao diện trực quan:

```bash
python app.py

```

* Truy cập ứng dụng tại: `http://localhost:5000` hoặc `http://127.0.0.1:5000`.


* Nhập một mã số **User ID** có sẵn từ bộ dữ liệu để hệ thống tự động lọc và gợi ý danh sách Top-10 bộ phim phù hợp nhất kèm điểm tin cậy đã được chuẩn hóa.



---

## 📊 Kết Quả Đạt Được

* 
**Hit Ratio (HR@10):** ~**79.99%** .


* 
**NDCG@10:** ~**0.5202**.


* 
**Catalog Coverage@10:** ~**37.38%** (Độ phủ rộng, gợi ý đa dạng phong phú).


* 
**Gini Index@10:** ~**0.8679** (Có thiên lệch nhẹ về phim phổ biến).
