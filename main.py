from functools import lru_cache
import os
import shutil
from uuid import uuid4
import faiss
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import torch
import uvicorn
from database import get_all_products, get_categories, get_product_info
from model_utils import get_model, transform
import torch.nn.functional as F
from math import ceil

app = FastAPI(title="Fashion ERP Product Search")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/fashion_resnet50.pth"
INDEX_PATH = "models/vector_db.index"
NAMES_PATH = "models/product_ids.npy"

model = get_model(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)
image_names = np.load(NAMES_PATH)


def _safe_upload_path(filename):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    return os.path.join("static/uploads", f"{uuid4().hex}{ext}")


def _extract_embedding(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        # THAY ĐỔI: Thêm chuẩn hóa F.normalize
        features = model(img_t).flatten()
        features = F.normalize(features, p=2, dim=0) 
        return features.cpu().numpy().astype("float32")

def _visual_signature(image):
    sample = image.convert("RGB").resize((96, 96))
    pixels = np.asarray(sample).reshape(-1, 3).astype("float32")
    max_channel = pixels.max(axis=1)
    min_channel = pixels.min(axis=1)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1)
    mask = (saturation > 0.08) & (max_channel < 248) & (min_channel > 5)
    focused = pixels[mask] if mask.any() else pixels
    color = focused.mean(axis=0) / 255.0

    gray = sample.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    gray_stat = ImageStat.Stat(gray)
    texture = np.array(
        [
            edge_stat.mean[0] / 255.0,
            edge_stat.stddev[0] / 128.0,
            gray_stat.stddev[0] / 128.0,
        ],
        dtype="float32",
    )
    return color.astype("float32"), texture


def _similarity_from_distance(distance, min_distance, max_distance):
    span = max(max_distance - min_distance, 1e-6)
    return 1.0 - ((distance - min_distance) / span)


def _color_similarity(a, b):
    return float(1.0 - min(np.linalg.norm(a - b) / np.sqrt(3), 1.0))


def _texture_similarity(a, b):
    return float(1.0 - min(np.linalg.norm(a - b) / np.sqrt(3), 1.0))


@lru_cache(maxsize=512)
def _product_signature(image_path):
    full_path = os.path.join("datasets", image_path.replace("/", os.sep))
    with Image.open(full_path).convert("RGB") as image:
        return _visual_signature(image)


def _rank_similar_products(image, top_k=9):
    query_vec = _extract_embedding(image)
    query_color, query_texture = _visual_signature(image)

    # Tăng search_k để lọc được nhiều ứng viên hơn trước khi xếp hạng
    search_k = min(index.ntotal, 100) 
    distances, indices = index.search(query_vec.reshape(1, -1), k=search_k)

    candidates = [
        (float(distance), int(idx))
        for distance, idx in zip(distances[0], indices[0])
        if idx >= 0
    ]
    if not candidates:
        return [], None

    category_votes = {}
    for distance, idx in candidates[:30]:
        info = get_product_info(image_names[idx])
        if not info:
            continue
        weight = 1.0 / (distance + 1e-6)
        category_votes[info["category"]] = category_votes.get(info["category"], 0.0) + weight
    predicted_category = max(category_votes, key=category_votes.get) if category_votes else None

    min_distance = min(distance for distance, _ in candidates)
    max_distance = max(distance for distance, _ in candidates)
    scored = []
    seen = set()

    # Tính toán lại min/max distance để chuẩn hóa
    valid_distances = [d for d, idx in zip(distances[0], indices[0]) if idx >= 0]
    min_dist = min(valid_distances) if valid_distances else 0
    max_dist = max(valid_distances) if valid_distances else 1

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        info = get_product_info(image_names[idx])
        if not info or info["id"] in seen: continue
        seen.add(info["id"])

        product_color, product_texture = _product_signature(info["image_path"])
        
        # 1. Điểm Vector (AI)
        vector_score = _similarity_from_distance(distance, min_dist, max_dist)
        
        # 2. Điểm Kiểu dáng (Rất quan trọng)
        category_score = 1.0 if predicted_category and info["category"] == predicted_category else 0.0
        
        # 3. Điểm Màu sắc (Quan trọng thứ 2)
        color_score = _color_similarity(query_color, product_color)
        
        # 4. Điểm Họa tiết (Dùng để phạt nếu áo trơn gặp áo họa tiết)
        texture_score = _texture_similarity(query_texture, product_texture)
        # Phạt nặng nếu texture lệch nhau (Ví dụ áo trơn vs áo hoa)
        texture_penalty = 1.0 if abs(query_texture[0] - product_texture[0]) < 0.15 else 0.5

        # THAY ĐỔI TRỌNG SỐ: Ưu tiên Style (Category) và Color
        
        # Tính giá trị cơ bản trước
        final_score = (
            0.35 * vector_score 
            + 0.35 * category_score 
            + 0.25 * color_score 
            + 0.05 * texture_score
        ) * texture_penalty

        # Sau đó mới áp dụng phạt nếu khác loại (trơn vs họa tiết)
        is_query_plain = query_texture[0] < 0.05 
        is_product_plain = product_texture[0] < 0.05

        if is_query_plain != is_product_plain:
            final_score *= 0.7

        item = dict(info)
        item["match_score"] = float(final_score) # Đảm bảo là float thường để JSON nhận diện
        # Tính % hiển thị thực tế hơn
        item["match"] = f"{int(round(final_score * 100))}%"
        scored.append(item)
    
    # Sắp xếp lại
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_k], predicted_category


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request, page: int = 1):
    all_products = get_all_products() # Giả sử hàm này trả về list toàn bộ sp
    
    # Logic phân trang
    items_per_page = 50
    total_items = len(all_products)
    total_pages = ceil(total_items / items_per_page)
    
    # Cắt mảng sản phẩm theo trang
    start = (page - 1) * items_per_page
    end = start + items_per_page
    products_to_show = all_products[start:end]
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "products": products_to_show,
            "categories": get_categories(),
            "current_page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
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
    os.makedirs("static/uploads", exist_ok=True)
    upload_path = _safe_upload_path(file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(upload_path).convert("RGB")
    results, _ = _rank_similar_products(image, top_k=9)

    return templates.TemplateResponse(
        request,
        "products.html",
        {
            "products": results,
            "query_img": os.path.basename(upload_path),
        },
    )


@app.post("/api/search-image")
async def search_image_api(file: UploadFile = File(...)):
    os.makedirs("static/uploads", exist_ok=True)
    upload_path = _safe_upload_path(file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(upload_path).convert("RGB")
    results, predicted_category = _rank_similar_products(image, top_k=9)

    # Sử dụng jsonable_encoder để bao bọc dữ liệu trả về
    return JSONResponse(
        content=jsonable_encoder({
            "status": "success",
            "message": f"Found {len(results)} matching products"
            + (f" in the {predicted_category} category" if predicted_category else ""),
            "results": results,
        })
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
