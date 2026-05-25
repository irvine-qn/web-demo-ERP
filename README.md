# Lumina Fashion ERP Search System

Lumina là demo ERP/e-commerce thời trang chạy bằng FastAPI, ResNet50 và FAISS. Dữ liệu sản phẩm hiện đọc từ `datasets/products.csv`; order và tồn kho demo được lưu in-memory để phục vụ quay video báo cáo.

## Demo Chính

- Storefront catalog: xem sản phẩm, lọc category, lọc giá, tìm kiếm text, sắp xếp và phân trang.
- Sales reservation: bấm `Buy Now` trên sản phẩm để tạo sales order và trừ ATP.
- Inventory dashboard: xem `Order History` và `Inventory Ledger`; trang này yêu cầu đăng nhập admin.
- Procurement image search: upload ảnh để tìm sản phẩm/supplier tương tự; Top 1 luôn `Out of Stock`, Top 2-10 là `Suggested Alternatives`.

## Tài Khoản Admin

Inventory yêu cầu đăng nhập:

```text
URL:      http://127.0.0.1:8000/admin/login
Username: admin
Password: 123456!
```

Procurement không yêu cầu đăng nhập để tiện demo:

```text
http://127.0.0.1:8000/admin/procurement
```

## Công Nghệ

- Backend: FastAPI, Python
- AI model: PyTorch, Torchvision, ResNet50
- Vector search: FAISS
- Image processing: Pillow
- Frontend: HTML, CSS, Vanilla JavaScript, Jinja2
- Data source: CSV metadata + FAISS index local

## Luồng Demo

### 1. Storefront & Sales Order

```text
http://127.0.0.1:8000/
```

1. Chọn hoặc tìm một sản phẩm trong catalog.
2. Bấm `Buy Now`.
3. Backend tạo order demo qua `POST /api/orders`.
4. ATP của SKU đó giảm 1.
5. Vào Inventory để xem order và thay đổi tồn kho.

### 2. Inventory Dashboard

```text
http://127.0.0.1:8000/admin/inventory
```

Trang gồm:

- `Order History`: danh sách sales order đã tạo.
- `Inventory Ledger`: SKU, ATP, reserved quantity và thời điểm cập nhật.

Trang tự refresh định kỳ để thể hiện stock reservation sau khi order được tạo.

### 3. Procurement Exception Handling

```text
http://127.0.0.1:8000/admin/procurement
```

1. Upload ảnh sản phẩm cần tìm nguồn hàng.
2. Backend dùng ResNet50 + FAISS để lấy Top 10 sản phẩm tương tự.
3. Top 1 được hardcode `ATP = 0`, hiển thị `Out of Stock` màu đỏ và ẩn nút order.
4. Top 2-10 được highlight với nhãn `Suggested Alternatives`.

## Mô Hình AI

ResNet50 đã fine-tune cho 6 category:

```text
Dress, Hat, Outerwear, Pant, Shirt, Shoes
```

Luồng xử lý ảnh:

1. Validate ảnh upload, giới hạn 12MB.
2. Resize/normalize ảnh theo cấu hình model.
3. Classifier dự đoán category.
4. Feature extractor sinh embedding 2048 chiều.
5. FAISS tìm ảnh gần nhất trong `models/vector_db.index`.
6. Backend kết hợp thêm màu sắc, texture và style key để ranking thực tế hơn.

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

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Chạy Server

```powershell
py -m uvicorn main:app --reload
```

Nếu port `8000` đang bận:

```powershell
py -m uvicorn main:app --reload --port 8001
```

## Tạo Lại FAISS Index

Chạy lại khi thay đổi dataset ảnh hoặc `products.csv`:

```powershell
py load_dataset.py
```

## API Đang Dùng

```text
GET  /
GET  /products
GET  /admin/login
POST /admin/login
GET  /admin/logout
GET  /admin/inventory
GET  /admin/procurement
POST /api/orders
GET  /api/orders
GET  /api/inventory
POST /api/procurement/search-image
```

Tạo order demo:

```json
{
  "product_id": "SP001",
  "quantity": 1
}
```

## Lưu Ý

- Order và inventory reservation là in-memory; restart server sẽ reset dữ liệu demo.
- Product catalog vẫn đọc từ `datasets/products.csv`.
- Ảnh upload được lưu tạm trong `static/uploads/`.
- Procurement Flow mở trực tiếp; Inventory Flow yêu cầu admin login.

Kiểm tra và giải phóng port trên Windows:

```powershell
Get-NetTCPConnection -LocalPort 8000
Stop-Process -Id <OwningProcess> -Force
```
