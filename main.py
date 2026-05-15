# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import random

app = FastAPI(title="Lumina Fashion")

os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- DATABASE SẢN PHẨM (Có gắn nhãn Màu sắc & Form dáng) ---
STORE_PRODUCTS = [
    # --- SHIRTS ---
    {"id": 1, "name": "Basic T-shirt", "price": "150,000₫", "category": "Shirt", "color": "White", "form": "Loose", "image": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=500&q=80"},
    {"id": 2, "name": "Elegant Shirt", "price": "250,000₫", "category": "Shirt", "color": "Black", "form": "Slim", "image": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=500&q=80"},
    {"id": 3, "name": "Winter Hoodie", "price": "350,000₫", "category": "Shirt", "color": "Yellow", "form": "Oversize", "image": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=500&q=80"},
    {"id": 4, "name": "Crop Sweater", "price": "220,000₫", "category": "Shirt", "color": "Red", "form": "Short", "image": "https://images.unsplash.com/photo-1613482184976-13d6a655ad1e?w=500&q=80"},
    # --- PANTS ---
    {"id": 5, "name": "Wide-leg Jeans", "price": "320,000₫", "category": "Pants", "color": "Blue", "form": "Loose", "image": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=500&q=80"},
    {"id": 6, "name": "Office Trousers", "price": "280,000₫", "category": "Pants", "color": "Black", "form": "Straight", "image": "https://images.unsplash.com/photo-1594938384824-03267591c28c?w=500&q=80"},
    {"id": 7, "name": "Active Shorts", "price": "180,000₫", "category": "Pants", "color": "Gray", "form": "Short", "image": "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=500&q=80"},
    {"id": 8, "name": "Cargo Kaki Pants", "price": "350,000₫", "category": "Pants", "color": "Brown", "form": "Loose", "image": "https://images.unsplash.com/photo-1555680202-c86f0e12f086?w=500&q=80"},
    # --- DRESSES ---
    {"id": 9, "name": "Pleated Skirt", "price": "190,000₫", "category": "Dress", "color": "White", "form": "A-line", "image": "https://images.unsplash.com/photo-1583496661160-c588c443c982?w=500&q=80"},
    {"id": 10, "name": "Party Dress", "price": "450,000₫", "category": "Dress", "color": "Red", "form": "Slim", "image": "https://images.unsplash.com/photo-1566150905458-1bf1fc113f0d?w=500&q=80"},
    {"id": 11, "name": "Floral Dress", "price": "260,000₫", "category": "Dress", "color": "Green", "form": "Flare", "image": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=500&q=80"},
    {"id": 12, "name": "Denim Skirt", "price": "210,000₫", "category": "Dress", "color": "Blue", "form": "A-line", "image": "https://images.unsplash.com/photo-1604085446654-71bc0e5dc7c7?w=500&q=80"},
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render giao diện chính có chứa các Tab"""
    return templates.TemplateResponse(request=request, name="index.html", context={"products": STORE_PRODUCTS})

@app.post("/api/search-image")
async def search_image(file: UploadFile = File(...)):
    """API giả lập AI: Nhận diện màu/form từ ảnh và tìm trong Database"""
    try:
        # 1. Lưu ảnh
        file_location = f"static/uploads/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 2. GIẢ LẬP KẾT QUẢ TỪ MODEL AI (Sau này Model thật sẽ trả về các thông số này)
        # Ở đây mình giả vờ AI nhận diện bức ảnh khách up lên là "Màu Đen" và "Form Rộng"
        ai_detected_color = random.choice(["Đen", "Trắng", "Đỏ", "Xanh dương"])
        ai_detected_form = random.choice(["Rộng", "Ôm", "Chữ A", "Suông"])
        
        # 3. Thuật toán tìm kiếm: Lọc trong STORE_PRODUCTS những món khớp màu hoặc form
        matched_items = []
        for p in STORE_PRODUCTS:
            score = 0
            if p["color"] == ai_detected_color: score += 50
            if p["form"] == ai_detected_form: score += 40
            
            # Nếu có điểm chung thì đưa vào kết quả
            if score > 0:
                matched_product = p.copy()
                matched_product["match"] = f"{score + random.randint(1, 9)}%" # Tạo số % cho sinh động
                matched_items.append(matched_product)
        
        # Sắp xếp từ giống nhất đến ít giống nhất
        matched_items.sort(key=lambda x: x["match"], reverse=True)

        return {
            "status": "success",
            "message": f"AI nhận diện: {ai_detected_color}, form {ai_detected_form}.",
            "results": matched_items[:4] # Trả về tối đa 4 món giống nhất
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}