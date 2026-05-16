from functools import lru_cache
import hashlib
import os
import re
from urllib.parse import urlencode
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
from model_utils import get_classifier_model, get_model, transform
from math import ceil

app = FastAPI(title="Fashion ERP Product Search")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/fashion_resnet50.pth"
INDEX_PATH = "models/vector_db.index"
NAMES_PATH = "models/product_ids.npy"

model = get_model(MODEL_PATH)
classifier_model = get_classifier_model(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)
image_names = np.load(NAMES_PATH)
CLASS_NAMES = ["Dress", "Hat", "Outerwear", "Pant", "Shirt", "Shoes"]
ITEMS_PER_PAGE = 50
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def _price_to_int(price):
    return int("".join(ch for ch in str(price) if ch.isdigit()) or 0)


def _filter_and_sort_products(products, category="all", sort_by="newest", query="", max_price=10000000):
    normalized_query = query.strip().lower()
    filtered = []
    for product in products:
        product_price = _price_to_int(product["price"])
        product_name = str(product["name"]).lower()
        if category != "all" and product["category"] != category:
            continue
        if normalized_query and normalized_query not in product_name:
            continue
        if product_price > max_price:
            continue
        item = dict(product)
        item["price_value"] = product_price
        filtered.append(item)

    if sort_by == "price-asc":
        filtered.sort(key=lambda item: item["price_value"])
    elif sort_by == "price-desc":
        filtered.sort(key=lambda item: item["price_value"], reverse=True)
    elif sort_by == "name":
        filtered.sort(key=lambda item: item["name"].lower())
    elif sort_by == "category":
        filtered.sort(key=lambda item: (item["category"], item["name"].lower()))
    else:
        filtered.sort(key=lambda item: item["id"], reverse=True)
    return filtered


def _pagination_prefix(category="all", sort="newest", q="", max_price=10000000):
    params = {}
    if category != "all":
        params["category"] = category
    if sort != "newest":
        params["sort"] = sort
    if q:
        params["q"] = q
    if max_price != 10000000:
        params["max_price"] = max_price
    query = urlencode(params)
    return f"/?{query}&" if query else "/?"


def _safe_upload_path(filename):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = ".jpg"
    return os.path.join("static/uploads", f"{uuid4().hex}{ext}")


def _is_supported_image(file):
    ext = os.path.splitext(file.filename or "")[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS and file.content_type in ALLOWED_IMAGE_TYPES


def _save_validated_upload(file):
    if not _is_supported_image(file):
        return None, "Định dạng file không hỗ trợ"

    os.makedirs("static/uploads", exist_ok=True)
    upload_path = _safe_upload_path(file.filename)
    total_size = 0
    with open(upload_path, "wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                buffer.close()
                os.remove(upload_path)
                return None, "Ảnh vượt quá dung lượng hỗ trợ 12MB"
            buffer.write(chunk)

    try:
        with Image.open(upload_path) as img:
            img.verify()
    except Exception:
        os.remove(upload_path)
        return None, "Định dạng file không hỗ trợ"

    return upload_path, None


def _image_digest(image):
    normalized = image.convert("RGB").resize((224, 224))
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _extract_embedding(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        # Keep query features in the same vector space used to build vector_db.index.
        return model(img_t).flatten().cpu().numpy().astype("float32")


def _predict_category(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = classifier_model(img_t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[top_idx], float(probs[top_idx].item())

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


def _style_key(name, category):
    text = str(name or "").lower()
    category_text = str(category or "").lower()
    text = re.sub(rf"\b{re.escape(category_text)}s?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _predict_style(category_candidates):
    style_votes = {}
    for distance, _, info in category_candidates[:24]:
        style = _style_key(info["name"], info["category"])
        if not style:
            continue
        weight = 1.0 / (distance + 1e-6)
        style_votes[style] = style_votes.get(style, 0.0) + weight
    if not style_votes:
        return None
    return max(style_votes, key=style_votes.get)


def _ranking_priority(style_match, color_score, shape_score, texture_score):
    shape_style_score = (0.82 * shape_score) + (0.18 * texture_score)
    color_match = color_score >= 0.78
    shape_match = shape_style_score >= 0.62
    pattern_match = texture_score >= 0.68

    if style_match and color_match and pattern_match:
        return 3, shape_style_score
    if shape_match and color_match:
        return 2, shape_style_score
    if color_match:
        return 1, shape_style_score
    return 0, shape_style_score


@lru_cache(maxsize=512)
def _product_signature(image_path):
    full_path = os.path.join("datasets", image_path.replace("/", os.sep))
    with Image.open(full_path).convert("RGB") as image:
        return _visual_signature(image)


@lru_cache(maxsize=512)
def _product_digest(image_path):
    full_path = os.path.join("datasets", image_path.replace("/", os.sep))
    with Image.open(full_path).convert("RGB") as image:
        return _image_digest(image)


def _absolute_distance_score(distance):
    if distance <= 1e-4:
        return 1.0
    if distance <= 80:
        return 0.9
    if distance <= 180:
        return 0.72
    if distance <= 300:
        return 0.5
    if distance <= 420:
        return 0.28
    return 0.16


def _is_out_of_domain(min_distance, category_confidence):
    return category_confidence < 0.45 and min_distance > 320


def _rank_similar_products(image, top_k=9):
    query_vec = _extract_embedding(image)
    query_color, query_texture = _visual_signature(image)
    query_digest = _image_digest(image)
    predicted_category, category_confidence = _predict_category(image)

    # Search the whole small catalog, then hard-filter by the classifier category.
    # This prevents a Hat query from being filled with Shoes just because they are
    # close in the embedding index.
    search_k = index.ntotal
    distances, indices = index.search(query_vec.reshape(1, -1), k=search_k)

    candidates = [
        (float(distance), int(idx))
        for distance, idx in zip(distances[0], indices[0])
        if idx >= 0
    ]
    if not candidates:
        return [], None

    nearest_distance = candidates[0][0]
    out_of_domain = _is_out_of_domain(nearest_distance, category_confidence)

    category_candidates = []
    fallback_candidates = []
    for distance, idx in candidates:
        info = get_product_info(image_names[idx])
        if not info:
            continue
        row = (distance, idx, info)
        if info["category"] == predicted_category:
            category_candidates.append(row)
        else:
            fallback_candidates.append(row)

    ranked_candidates = [] if out_of_domain else category_candidates
    if len(ranked_candidates) < top_k:
        ranked_candidates = category_candidates + fallback_candidates
    predicted_style = _predict_style(category_candidates)

    valid_distances = [distance for distance, _, _ in ranked_candidates]
    min_dist = min(valid_distances) if valid_distances else 0
    max_dist = max(valid_distances) if valid_distances else 1
    scored = []
    seen = set()

    for distance, idx, info in ranked_candidates:
        if not info or info["id"] in seen:
            continue
        seen.add(info["id"])

        product_color, product_texture = _product_signature(info["image_path"])
        exact_image_match = query_digest == _product_digest(info["image_path"])

        # 1. Shape/form score from ResNet feature distance.
        vector_score = _similarity_from_distance(distance, min_dist, max_dist)
        absolute_score = _absolute_distance_score(distance)

        # 2. Category is already hard-filtered when enough products exist.
        category_score = 1.0 if info["category"] == predicted_category else 0.0

        # 3. Color is the second gate after exact shape/style+color matches.
        color_score = _color_similarity(query_color, product_color)

        # 4. Texture/pattern separates plain products from patterned ones.
        texture_score = _texture_similarity(query_texture, product_texture)
        texture_penalty = 1.0 if abs(query_texture[0] - product_texture[0]) < 0.15 else 0.5
        product_style = _style_key(info["name"], info["category"])
        style_match = bool(predicted_style and product_style == predicted_style)
        priority, shape_style_score = _ranking_priority(
            style_match,
            color_score,
            vector_score,
            texture_score,
        )

        # Tăng độ khớp cho ảnh giống hệt (vector_score, color_score, texture_score đều rất cao)
        # Nếu giống hoàn toàn (tất cả đều > 0.98), match = 99%
        if exact_image_match and category_score == 1.0:
            final_score = 0.99
        else:
            # Nếu khác màu rõ rệt hoặc chỉ hơi giống kiểu dáng, giảm mạnh tỉ lệ khớp
            # Nếu color_score < 0.5 hoặc vector_score < 0.5 thì match < 40%
            base_score = (
                0.46 * shape_style_score
                + 0.32 * color_score
                + 0.17 * texture_score
                + 0.05 * category_score
            ) * texture_penalty * (0.65 + 0.35 * absolute_score)

            is_query_plain = query_texture[0] < 0.05
            is_product_plain = product_texture[0] < 0.05

            if is_query_plain != is_product_plain:
                base_score *= 0.76

            priority_bands = {
                3: (0.72, 0.16),
                2: (0.56, 0.13),
                1: (0.42, 0.11),
                0: (0.24, 0.13),
            }
            band_min, band_width = priority_bands[priority]
            final_score = band_min + (band_width * base_score)

            # Nếu khác màu rõ rệt hoặc chỉ hơi giống kiểu dáng, giảm mạnh tỉ lệ khớp
            if color_score < 0.5 or vector_score < 0.5:
                final_score = min(final_score, 0.39)
            if out_of_domain:
                final_score = min(final_score, 0.29)
            elif absolute_score < 0.35 and category_confidence < 0.6:
                final_score = min(final_score, 0.38)

        item = dict(info)
        item["match_score"] = float(final_score)
        item["priority"] = priority
        item["color_score"] = color_score
        item["shape_score"] = shape_style_score
        item["detected_category"] = predicted_category
        item["category_confidence"] = round(category_confidence * 100, 1)
        item["detected_style"] = predicted_style
        item["detected_gender_style"] = item.get("gender_style")
        # Tính % hiển thị thực tế hơn
        item["match"] = f"{int(round(max(1, min(99, final_score * 100))))}%"
        scored.append(item)
    
    # Sắp xếp giảm dần theo match_score (tỉ lệ khớp), nếu bằng thì ưu tiên priority, shape_score, color_score
    scored.sort(
        key=lambda x: (
            x["match_score"],
            x["priority"],
            x["shape_score"],
            x["color_score"],
        ),
        reverse=True,
    )
    if not out_of_domain and len(category_candidates) >= top_k:
        scored = [item for item in scored if item["category"] == predicted_category]
    for item in scored:
        item.pop("match_score", None)
        item.pop("priority", None)
        item.pop("color_score", None)
        item.pop("shape_score", None)
    return scored[:top_k], predicted_category


@app.get("/", response_class=HTMLResponse)
async def index_page(
    request: Request,
    page: int = 1,
    category: str = "all",
    sort: str = "newest",
    q: str = "",
    max_price: int = 10000000,
):
    all_products = get_all_products() # Giả sử hàm này trả về list toàn bộ sp
    
    # Logic phân trang
    filtered_products = _filter_and_sort_products(
        all_products,
        category=category,
        sort_by=sort,
        query=q,
        max_price=max_price,
    )
    total_items = len(filtered_products)
    total_pages = max(1, ceil(total_items / ITEMS_PER_PAGE))
    page = max(1, min(page, total_pages))
    
    # Cắt mảng sản phẩm theo trang
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    products_to_show = filtered_products[start:end]
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "products": products_to_show,
            "categories": get_categories(),
            "current_page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "total_items": total_items,
            "all_items_count": len(all_products),
            "active_category": category,
            "active_sort": sort,
            "search_query": q,
            "max_price": max_price,
            "page_url_prefix": _pagination_prefix(category, sort, q, max_price),
        },
    )


@app.get("/products", response_class=HTMLResponse)
async def products_page(
    request: Request,
    page: int = 1,
    category: str = "all",
    sort: str = "newest",
    q: str = "",
    max_price: int = 10000000,
):
    return await index_page(request, page, category, sort, q, max_price)


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    upload_path, error = _save_validated_upload(file)
    if error:
        return templates.TemplateResponse(
            request,
            "products.html",
            {"products": [], "query_img": None, "error": error},
            status_code=400,
        )

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
    upload_path, error = _save_validated_upload(file)
    if error:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": error, "results": []},
        )

    image = Image.open(upload_path).convert("RGB")

    results, predicted_category = _rank_similar_products(image, top_k=9)
    # Nếu không có sản phẩm hoặc out-of-domain (ảnh không phải thời trang)
    if not results or (predicted_category is None or all(float(item.get("category_confidence", 0)) < 45 for item in results)):
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "status": "no-fashion",
                "message": "We cannot find any products matching this in our fashion category.",
                "results": [],
            })
        )
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
