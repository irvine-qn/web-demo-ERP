# TOPIC: Utilizing Image Processing Model into ERP Systems for Product Searching in Fashion E-Commerce Platform
# Project Name: Lumina Fashion ERP Search System

## Tổng Quan Dự Án

LUMINA là demo ERP/e-commerce cho thời trang, tập trung vào tìm kiếm sản phẩm bằng hình ảnh. Người dùng có thể upload, kéo thả hoặc paste ảnh trang phục; hệ thống phân tích ảnh và trả về các sản phẩm có kiểu dáng, màu sắc và họa tiết gần nhất trong catalog.

## Tính Năng Chính

- AI Image Search: tìm sản phẩm bằng ảnh từ file, drag/drop hoặc clipboard.
- Visual Ranking: ưu tiên kết quả theo thứ tự `kiểu dáng + màu sắc`, sau đó `màu sắc`, rồi mới đến `kiểu dáng`.
- Match Score thực tế: ảnh trùng dataset được giới hạn tối đa 99%, không hiển thị 100% vì ảnh có thể bị nén, mờ, resize hoặc sai khác nhỏ.
- Category Filtering: lọc theo danh mục sản phẩm và khoảng giá.
- Sorting/Pagination: sắp xếp theo giá, tên, danh mục; phân trang 50 sản phẩm/trang.
- Responsive UI: desktop 5 cột, tablet 3 cột, mobile 2 cột; kết quả AI giữ layout riêng.
- Gender Style Classification: gán nhãn phong cách `Masculine`, `Feminine`, `Neutral` cho sản phẩm thời trang.
- Edge Case Handling: chặn file không hỗ trợ, giới hạn upload 12MB, giảm điểm ảnh ngoài domain thời trang, hiển thị `No products found` khi database rỗng.

## Công Nghệ Sử Dụng

- Backend: FastAPI, Python.
- AI/Deep Learning: PyTorch, Torchvision.
- Vector Search: FAISS.
- Image Processing: ResNet50, Pillow.
- Frontend: HTML5, CSS3, Vanilla JavaScript.
- Database: CSV metadata và FAISS index.

## Chi Tiết Mô Hình AI

Hệ thống dùng ResNet50 đã fine-tune cho 6 nhóm sản phẩm: `Dress`, `Hat`, `Outerwear`, `Pant`, `Shirt`, `Shoes`.

Luồng hoạt động:

1. Ảnh upload được resize/normalize giống lúc train.
2. ResNet50 classifier dự đoán category sản phẩm.
3. Feature extractor lấy embedding 2048 chiều từ ResNet50.
4. FAISS tìm các sản phẩm gần nhất trong vector database.
5. Backend tính thêm chữ ký thị giác bằng Pillow:
   - màu chủ đạo từ RGB mean có lọc saturation,
   - texture từ edge/contrast để phân biệt trơn và họa tiết,
   - style key từ tên sản phẩm trong `products.csv`.
6. Ranking cuối cùng ưu tiên:
   - category/shape đúng,
   - màu giống,
   - texture/họa tiết giống,
   - style name giống.

Ảnh không phải thời trang thường có confidence category thấp và khoảng cách FAISS cao, nên điểm match bị giới hạn thấp thay vì bị đẩy lên cao bởi chuẩn hóa tương đối.

## Fine-Tune Model

ResNet50 được fine-tune bằng transfer learning:

1. Dùng backbone ResNet50 pre-trained.
2. Thay fully connected layer cuối thành 6 output class theo category thời trang.
3. Train trên dataset ảnh chia theo thư mục category.
4. Lưu trọng số vào `models/fashion_resnet50.pth`.
5. Khi tạo index, bỏ classifier head và dùng phần backbone để sinh embedding 2048 chiều cho từng ảnh.

Phân loại giới tính thời trang hiện được triển khai ở tầng metadata/rule-based vì `products.csv` chưa có nhãn gender train riêng. Hệ thống trả về 3 nhãn:

- `Masculine`: các item có keyword như Henley, Oxford, Polo, Cargo, Chino, Derby, Loafer, Bomber.
- `Feminine`: Dress hoặc keyword như Floral, Blouse, Gown, Heels, Sandal.
- `Neutral`: các sản phẩm còn lại hoặc sản phẩm có phong cách unisex.

Nếu muốn fine-tune gender model thật sự, cần bổ sung nhãn `gender_style` vào dataset và train thêm classifier 3 lớp: `Masculine`, `Feminine`, `Neutral`.

## Vì Sao Chọn ResNet50

- Đủ mạnh cho trích xuất đặc trưng hình ảnh thời trang nhưng vẫn chạy được trong demo local.
- Có backbone pre-trained ổn định, dễ fine-tune với dataset vừa và nhỏ.
- Embedding 2048 chiều phù hợp với FAISS để tìm kiếm nhanh.
- Dễ debug hơn các kiến trúc lớn hơn như ViT/CLIP khi mục tiêu demo là category + visual similarity trong catalog nội bộ.

## Cấu Trúc Thư Mục

```text
WEB DEMO ERP/
├── models/
│   ├── fashion_resnet50.pth
│   ├── vector_db.index
│   └── product_ids.npy
├── datasets/
│   ├── Dress/, Shirt/, Pant/, ...
│   └── products.csv
├── static/
│   ├── css/style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   └── products.html
├── database.py
├── model_utils.py
├── load_dataset.py
├── main.py
└── requirements.txt
```

## Cài Đặt Và Chạy

Yêu cầu Python 3.9 trở lên.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu thiếu thư viện AI:

```bash
pip install torch torchvision faiss-cpu
```

Tạo lại FAISS index sau khi thay dataset:

```bash
python load_dataset.py
```

Chạy server:

```bash
py -m uvicorn main:app --reload
```

Truy cập: http://127.0.0.1:8000

## Test Case Đã Xử Lý

- Ảnh có trong dataset: kết quả đúng được ưu tiên top 1, match tối đa 99%.
- Ảnh quần áo ngoài dataset: tìm sản phẩm cùng vibe/category thay vì yêu cầu match tuyệt đối.
- Ảnh bị mờ/resize: vẫn dùng shape, color, texture để ranking.
- Ảnh không phải thời trang: match bị hạ thấp khi confidence thấp và khoảng cách FAISS cao.
- File `.txt`, `.pdf` hoặc định dạng không hỗ trợ: trả lỗi `Định dạng file không hỗ trợ`.
- `products.csv` rỗng: giao diện hiển thị `No products found`, không trắng trang.

---
*Project Name: Lumina Fashion ERP Search System*

*Author: Vo Anh Hao, Irvine*

