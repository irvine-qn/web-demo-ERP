Utilizing Image Processing Model into ERP Systems for Product Searching in Fashion E-Commerce Platform

📌 Tổng quan dự án (Project Overview)

Dự án LUMINA là một hệ thống ERP (Enterprise Resource Planning) hiện đại dành
cho thương mại điện tử thời trang. Điểm đột phá của hệ thống là việc tích hợp mô
hình xử lý hình ảnh (Image Processing) cho phép người dùng tìm kiếm sản phẩm
thông qua hình ảnh thay vì từ khóa truyền thống. Hệ thống giúp tối ưu hóa trải
nghiệm khách hàng và quản lý kho hàng thông minh dựa trên đặc trưng thị giác
(Visual Features).

🚀 Tính năng chính (Key Features)

  - AI Image Search: Tìm kiếm sản phẩm bằng cách upload ảnh, kéo thả hoặc dán
    ảnh từ clipboard.
  - Visual Ranking: Hệ thống phân loại và sắp xếp kết quả theo độ khớp (%) dựa
    trên Kiểu dáng (Style), Màu sắc (Color) và Họa tiết (Texture).
  - ERP Product Management: Trang chủ hiển thị danh sách sản phẩm theo dạng Grid
    (5 cột), hỗ trợ phân trang (Pagination) 50 sản phẩm/trang.
  - Responsive Design: Giao diện tối ưu cho Desktop (5 cột), Tablet (3 cột) và
    Mobile (2 cột).
  - Category Filtering: Lọc sản phẩm theo danh mục và khoảng giá.

🛠 Công nghệ sử dụng (Tech Stack)

  - Backend: FastAPI (Python) - Hiệu năng cao, xử lý bất đồng bộ.
  - AI/Deep Learning: PyTorch, Torchvision.
  - Vector Search: FAISS (Facebook AI Similarity Search).
  - Image Processing: OpenCV, Pillow (PIL).
  - Frontend: HTML5, CSS3 (Lumina Design System), JavaScript (Vanilla JS).
  - Database: CSV (Lưu trữ metadata sản phẩm) & .index (Lưu trữ vector AI).

🧠 Chi tiết mô hình AI (Model Details)

Hệ thống sử dụng phương pháp Feature Fusion (Kết hợp đặc trưng) để đạt độ chính
xác cao nhất:

1.  Kiểu dáng (Style): Sử dụng mô hình ResNet50 đã qua huấn luyện (Pre-trained
    on ImageNet/Fine-tuned) để trích xuất Embedding Vector (2048 chiều).
2.  Màu sắc (Color): Sử dụng Color Histogram trong không gian màu RGB để nhận
    diện màu sắc chủ đạo.
3.  Họa tiết (Texture): Phân tích độ tương phản và cạnh (Edges) để phân biệt đồ
    trơn và đồ có họa tiết.
4.  Ranking: Kết quả được tính toán bằng Cosine Similarity và trọng số ưu tiên:
    Style (46%) + Category (28%) + Color (18%) + Texture (8%).

📁 Cấu trúc thư mục (Directory Structure)

WEB DEMO ERP/
├── models/                     # Chứa file mô hình và index AI
│   ├── fashion_resnet50.pth    # Trọng số mô hình ResNet50
│   ├── vector_db.index         # Danh bạ vector AI (FAISS)
│   └── product_ids.npy         # Mapping ID sản phẩm
├── datasets/                   # Kho dữ liệu ERP
│   ├── Dress/, Shirt/, ...     # Thư mục ảnh theo phân loại
│   └── products.csv            # Metadata sản phẩm (ID, Name, Price...)
├── static/                     # Tài nguyên tĩnh
│   ├── css/style.css           # Giao diện Lumina Design
│   └── uploads/                # Lưu trữ ảnh user search tạm thời
├── templates/                  # Giao diện HTML (Jinja2)
│   ├── index.html              # Trang chủ & AI Search
│   └── products.html           # Trang kết quả tìm kiếm
├── database.py                 # Xử lý truy vấn CSV
├── model_utils.py              # Cấu hình PyTorch Model
├── load_dataset.py             # Script tạo Index AI (Run first)
├── main.py                     # Server Backend FastAPI
└── requirements.txt            # Danh sách thư viện cần cài đặt

🛠 Hướng dẫn cài đặt & Chạy (Installation & Setup)

1. Cài đặt môi trường

Yêu cầu Python 3.9 trở lên.

# Clone dự án
git clone https://github.com/irvine-qn/web-demo-ERP
cd fashion-erp-ai

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

* Cài thêm những nội dung sau nếu `pip install -r requirements.txt` không thể cài đặt hết được:
pip install torch torchvision
pip install faiss-cpu

2. Chuẩn bị dữ liệu AI (Quan trọng)

Trước khi chạy website, bạn cần tạo file danh bạ vector (.index) từ kho ảnh sản
phẩm:

1.  Đảm bảo ảnh sản phẩm đã nằm trong thư mục datasets/.
2.  Đảm bảo file models/fashion_resnet50.pth đã sẵn sàng.
3.  Chạy script tạo index:

python load_dataset.py

3. Khởi chạy ứng dụng

Chạy server FastAPI bằng Uvicorn:

py -m uvicorn main:app --reload

Truy cập ứng dụng tại: http://127.0.0.1:8000

📖 Hướng dẫn sử dụng (Usage)

1.  Xem sản phẩm: Trang chủ sẽ liệt kê toàn bộ sản phẩm từ ERP, sử dụng phân
    trang ở dưới cùng.
2.  Tìm kiếm bằng hình ảnh:
      - Chuyển sang tab AI Image Search.
      - Kéo một ảnh từ máy tính hoặc dán (Ctrl+V) ảnh một chiếc áo/quần vào vùng
        upload.
      - Bấm "ANALYZE WITH AI".
      - Hệ thống sẽ trả về Top 9 sản phẩm giống nhất kèm độ khớp (Match %).

Project Name: Lumina Fashion ERP Search System
Author: Irvine
Topic: Utilizing Image Processing Model into ERP Systems.
