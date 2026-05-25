# Lumina Fashion ERP Search System

Demo ERP/e-commerce cho ngành thời trang, kết hợp FastAPI, ResNet50 và FAISS để tìm sản phẩm bằng hình ảnh. Hệ thống hiện dùng file CSV làm nguồn dữ liệu chính, phù hợp để chạy demo local và quay video báo cáo.

## Tính Năng Chính

- Storefront catalog: xem danh sách sản phẩm, lọc theo category, khoảng giá, tìm kiếm text, sắp xếp và phân trang.
- AI Image Search: upload, kéo thả hoặc paste ảnh để tìm sản phẩm tương tự.
- Visual ranking: kết hợp category, form dáng, màu sắc, texture và style name để xếp hạng kết quả.
- ATP badge: kết quả AI hiển thị số lượng tồn khả dụng `ATP`.
- Sales flow: nút `Order Now` tạo đơn hàng demo và tự động trừ ATP.
- Admin Inventory Dashboard: xem `Order History` và `Inventory Ledger`, tự refresh để chứng minh stock reservation.
- Procurement flow: upload ảnh tìm nguồn hàng supplier, hardcode Top 1 `Out of Stock`, highlight Top 2-4 là `Suggested Alternatives`.
- Admin login: bảo vệ các trang admin bằng tài khoản test.

## Tài Khoản Admin Test

Truy cập:

```text
http://127.0.0.1:8000/admin/login
```

Thông tin đăng nhập:

```text
Username: admin
Password: 123456!
```

## Công Nghệ Sử Dụng

- Backend: FastAPI, Python
- AI/Deep Learning: PyTorch, Torchvision
- Feature extractor/classifier: ResNet50
- Vector search: FAISS
- Image processing: Pillow
- Frontend: HTML, CSS, Vanilla JavaScript, Jinja2 templates
- Data source: `datasets/products.csv` và FAISS index local

## Luồng Demo

### 1. Sales Flow - Storefront

URL:

```text
http://127.0.0.1:8000/
```

Các bước demo:

1. Mở tab `AI Image Search`.
2. Upload một ảnh sản phẩm thời trang.
3. Backend giả lập thời gian xử lý khoảng 1-2 giây.
4. Kết quả trả về dạng grid sản phẩm, mỗi sản phẩm có badge `ATP: <số lượng>`.
5. Bấm `Order Now`.
6. API tạo đơn hàng mới và giảm ATP của SKU đó đi 1.

API liên quan:

```text
POST /api/search-image
POST /api/orders
```

### 2. Inventory Flow - Admin Dashboard

URL:

```text
http://127.0.0.1:8000/admin/inventory
```

Trang này gồm:

- `Order History`: hiển thị đơn hàng vừa tạo từ Storefront.
- `Inventory Ledger`: hiển thị danh sách SKU, ATP và reserved quantity.

Khi bấm `Order Now` ở Storefront, bảng inventory sẽ tự refresh và số lượng ATP giảm đi 1.

API liên quan:

```text
GET /api/orders
GET /api/inventory
```

### 3. Procurement Flow & Exception Handling

URL:

```text
http://127.0.0.1:8000/admin/procurement
```

Các bước demo:

1. Upload ảnh sản phẩm để tìm nguồn hàng supplier.
2. Backend trả về danh sách sản phẩm tương tự.
3. Top 1 luôn được hardcode `ATP = 0` và hiển thị `Out of Stock` màu đỏ.
4. Nút order của Top 1 bị ẩn.
5. Top 2, Top 3, Top 4 được highlight bằng nhãn `Suggested Alternatives`.

API liên quan:

```text
POST /api/procurement/search-image
```

## Mô Hình AI

Hệ thống dùng ResNet50 đã fine-tune cho 6 nhóm sản phẩm:

```text
Dress, Hat, Outerwear, Pant, Shirt, Shoes
```

Luồng xử lý ảnh:

1. Ảnh upload được validate định dạng và giới hạn dung lượng 12MB.
2. Ảnh được resize/normalize theo cấu hình train.
3. Classifier dự đoán category sản phẩm.
4. Feature extractor sinh embedding 2048 chiều.
5. FAISS tìm các ảnh gần nhất trong `models/vector_db.index`.
6. Backend tính thêm màu sắc, texture và style key để ranking thực tế hơn.
7. Kết quả trả về kèm match score và ATP.

## Cấu Trúc Thư Mục

```text
web demo ERP/
├── datasets/
│   ├── Dress/, Hat/, Outerwear/, Pant/, Shirt/, Shoes/
│   └── products.csv
├── models/
│   ├── fashion_resnet50.pth
│   ├── vector_db.index
│   └── product_ids.npy
├── static/
│   ├── css/style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   ├── products.html
│   ├── admin_inventory.html
│   ├── admin_login.html
│   └── admin_procurement.html
├── database.py
├── load_dataset.py
├── main.py
├── model_utils.py
└── requirements.txt
```

## Cài Đặt

Yêu cầu Python 3.9 trở lên.

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Nếu không dùng virtualenv, có thể cài trực tiếp bằng:

```powershell
py -m pip install -r requirements.txt
```

## Chạy Server

Chạy ở port mặc định `8000`:

```powershell
py -m uvicorn main:app --reload
```

Hoặc chạy port khác nếu `8000` đang bận:

```powershell
py -m uvicorn main:app --reload --port 8001
```

Truy cập:

```text
http://127.0.0.1:8000
```

Nếu dùng port `8001`:

```text
http://127.0.0.1:8001
```

## Tạo Lại FAISS Index

Chạy lệnh này khi thay đổi dataset ảnh hoặc metadata:

```powershell
py load_dataset.py
```

Các file index được tạo/làm mới trong thư mục `models/`:

```text
models/vector_db.index
models/product_ids.npy
```

## API Chính

```text
GET  /
GET  /products
GET  /admin/login
POST /admin/login
GET  /admin/logout
GET  /admin/inventory
GET  /admin/procurement
POST /api/search-image
POST /api/orders
GET  /api/orders
GET  /api/inventory
POST /api/procurement/search-image
```

Tạo order demo:

```json
POST /api/orders
{
  "product_id": "SP001",
  "quantity": 1
}
```

## Lưu Ý Demo

- Order và inventory reservation đang lưu in-memory để phục vụ demo. Khi restart server, order history và ATP reservation sẽ reset.
- Metadata sản phẩm vẫn đọc từ `datasets/products.csv`.
- Ảnh upload được lưu tạm trong `static/uploads/`.
- Nếu gặp lỗi port bị chiếm, dùng port khác hoặc dừng process đang giữ port.

Kiểm tra port trên Windows:

```powershell
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <OwningProcess> -Force
```

## Test Case Đã Xử Lý

- Upload ảnh đúng định dạng: trả về sản phẩm tương tự.
- File không phải ảnh hoặc sai định dạng: trả lỗi validate.
- Ảnh vượt quá 12MB: bị từ chối.
- Ảnh ngoài domain thời trang: trả thông báo không tìm thấy sản phẩm phù hợp.
- Tạo order thành công: ATP giảm 1 và order xuất hiện trong admin inventory.
- Procurement Top 1 hết hàng: ẩn nút order và gợi ý alternatives.

---

Project: Lumina Fashion ERP Search System