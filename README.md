TỔNG QUAN CẤU TRÚC THƯ MỤC

```
WEB DEMO ERP/
│
├── .venv/                      # Môi trường ảo (Chứa thư viện torch, flask, faiss...)
│
├── datasets/                   # KHO DỮ LIỆU GỐC
│   ├── Dress/                  # Ảnh các loại váy
│   ├── Hat/                    # Ảnh các loại mũ
│   ├── Outerwear/              # Ảnh áo khoác
│   ├── Pant/                   # Ảnh quần
│   ├── Shirt/                  # Ảnh áo sơ mi
│   ├── Shoes/                  # Ảnh giày dép
│   └── products.csv            # File chứa thông tin chi tiết (ID, Tên, Giá, Mô tả)
│
├── models/                     # KHO CHỨA "BỘ NÃO" AI (Tải từ Colab về)
│   ├── ResNet50_Fashion.ipynb  # File lưu trữ code colab (để backup)
│   ├── fashion_resnet50.pth    # FILE 1: Trọng số mô hình đã train
│   ├── vector_db.index         # FILE 2: Danh bạ vector của toàn bộ ảnh trong dataset
│   └── product_ids.npy         # FILE 3: Danh sách tên file ảnh tương ứng với vector
│
├── static/                     # TÀI NGUYÊN TĨNH CHO WEBSITE
│   ├── css/
│   │   └── style.css           # Giao diện cho Website
│   └── uploads/                # Nơi lưu ảnh mà USER upload lên để tìm kiếm
│
├── templates/                  # GIAO DIỆN HTML (Flask/FastAPI)
│   ├── index.html              # Trang chủ (nơi có nút upload ảnh)
│   └── products.html           # Trang hiển thị danh sách sản phẩm tìm được
│
├── database.py                 # Xử lý Logic Database (Đọc file products.csv)
├── load_dataset.py             # Script bổ trợ để load mô hình hoặc cập nhật index
├── main.py                     # FILE CHẠY CHÍNH (Server Backend)
└── requirements.txt            # Danh sách thư viện cần cài (torch, flask, faiss...)
```

# Hướng dẫn cài đặt và chạy code
- Sử dụng .venv
- Chạy terminal

- Cài đặt requirement.txt: dán nội dung sau vào terminal
pip install -r requirements.txt
pip install torch torchvision
pip install faiss-cpu

- Chạy code: 
py -m uvicorn main:app --reload