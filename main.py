from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
import torch
import faiss
import numpy as np
from PIL import Image

from model_utils import get_model, transform
from database import get_all_products, get_categories, get_product_info

app = FastAPI(title="Fashion ERP Product Search")

# 1. Cấu hình các thư mục tĩnh và giao diện
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets") # Để hiển thị ảnh sản phẩm
templates = Jinja2Templates(directory="templates")

# 2. Load AI Model và FAISS Index
MODEL_PATH = 'models/fashion_resnet50.pth'
INDEX_PATH = 'models/vector_db.index'
NAMES_PATH = 'models/product_ids.npy'

model = get_model(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)
image_names = np.load(NAMES_PATH)

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "products": get_all_products(),
            "categories": get_categories(),
        },
    )

@app.get("/products", response_class=HTMLResponse)
async def products_page(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "products": get_all_products(),
            "categories": get_categories(),
        },
    )

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    # 1. Lưu ảnh người dùng upload
    os.makedirs("static/uploads", exist_ok=True)
    upload_path = os.path.join("static/uploads", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Xử lý ảnh bằng AI
    image = Image.open(upload_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        query_vec = model(img_t).flatten().numpy()

    # 3. Tìm kiếm trong FAISS (lấy top 5)
    D, I = index.search(query_vec.reshape(1, -1), k=5)

    # 4. Lấy thông tin chi tiết từ Database
    results = []
    for idx in I[0]:
        img_name = image_names[idx]
        info = get_product_info(img_name)
        if info:
            results.append(info)

    # 5. Trả về giao diện kèm kết quả
    return templates.TemplateResponse(
        request,
        "products.html",
        {
            "products": results,
            "query_img": file.filename
        }
    )

@app.post("/api/search-image")
async def search_image_api(file: UploadFile = File(...)):
    os.makedirs("static/uploads", exist_ok=True)
    upload_path = os.path.join("static/uploads", file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(upload_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        query_vec = model(img_t).flatten().numpy()

    distances, indices = index.search(query_vec.reshape(1, -1), k=5)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        img_name = image_names[idx]
        info = get_product_info(img_name)
        if info:
            info["match"] = f"{max(0, min(99, 99 - int(distance)))}%"
            results.append(info)

    return JSONResponse(
        {
            "status": "success",
            "message": f"Tim thay {len(results)} san pham tu database",
            "results": results,
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
