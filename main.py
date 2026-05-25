import asyncio
from datetime import datetime
from functools import lru_cache
import hashlib
import os
import re
from urllib.parse import urlencode
from uuid import uuid4
import faiss
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import numpy as np
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn
from database import get_all_products, get_categories, get_product_by_id, get_product_info
from feature_extractors import (
    FUSED_DIM,
    build_combined_vector,
    cosine_similarity,
    extract_color_histogram,
    extract_hog_features,
)
from model_utils import get_classifier_model, get_model, transform
from math import ceil

app = FastAPI(title="Fashion ERP Product Search")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/fashion_resnet50.pth"
INDEX_PATH = "models/vector_db.index"
NAMES_PATH = "models/product_ids.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(MODEL_PATH).to(device)
classifier_model = get_classifier_model(MODEL_PATH).to(device)
index = faiss.read_index(INDEX_PATH)
if index.d != FUSED_DIM:
    raise RuntimeError(
        f"FAISS index dimension ({index.d}) != fused features ({FUSED_DIM}). "
        "Rebuild index: python load_dataset.py"
    )
image_names = np.load(NAMES_PATH)
CLASS_NAMES = ["Dress", "Hat", "Outerwear", "Pant", "Shirt", "Shoes"]
ITEMS_PER_PAGE = 50
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ADMIN_COOKIE_NAME = "lumina_admin"
ADMIN_ACCOUNTS = {
    "admin": "123456!",
}


class OrderCreateRequest(BaseModel):
    product_id: str
    quantity: int = 1


ORDERS = []
INVENTORY_LEDGER = {}


def _initial_atp(product_id):
    digits = int("".join(ch for ch in str(product_id) if ch.isdigit()) or 1)
    return 6 + (digits % 12)


def _get_product_by_id(product_id):
    return get_product_by_id(product_id)


def _ensure_inventory():
    if INVENTORY_LEDGER:
        return
    for product in get_all_products():
        INVENTORY_LEDGER[str(product["id"])] = {
            "sku": str(product["id"]),
            "name": product["name"],
            "category": product["category"],
            "price": product["price"],
            "image": product["image"],
            "atp": _initial_atp(product["id"]),
            "reserved": 0,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def _get_atp(product_id):
    _ensure_inventory()
    item = INVENTORY_LEDGER.get(str(product_id))
    return item["atp"] if item else 0


def _with_inventory(product):
    item = dict(product)
    item["atp"] = int(_get_atp(str(item["id"])))
    return item


def _enrich_customer_search_results(results):
    """Attach real ATP for storefront image search (no procurement exceptions)."""
    _ensure_inventory()
    enriched = []
    for rank, item in enumerate(results, start=1):
        row = dict(item)
        product_id = str(row["id"])
        atp = int(_get_atp(product_id))
        row["atp"] = atp
        row["rank"] = rank
        row["orderable"] = atp > 0
        row["availability_label"] = f"ATP: {atp}" if atp > 0 else "Out of Stock"
        enriched.append(row)
    return enriched


def _inventory_rows():
    _ensure_inventory()
    rows = list(INVENTORY_LEDGER.values())
    rows.sort(key=lambda item: item["sku"])
    return rows


def _is_admin_logged_in(request):
    return request.cookies.get(ADMIN_COOKIE_NAME) == "authenticated"


def _admin_login_redirect():
    return RedirectResponse(url="/admin/login", status_code=303)


def _admin_portal_redirect():
    return RedirectResponse(url="/admin", status_code=303)


def _set_admin_session(response):
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value="authenticated",
        httponly=True,
        samesite="lax",
    )
    return response


def _verify_admin_credentials(username, password):
    stored = ADMIN_ACCOUNTS.get(str(username).strip())
    return stored is not None and stored == password


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
        return None, "Dinh dang file khong ho tro"

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
                return None, "Anh vuot qua dung luong ho tro 12MB"
            buffer.write(chunk)

    try:
        with Image.open(upload_path) as img:
            img.verify()
    except Exception:
        os.remove(upload_path)
        return None, "Dinh dang file khong ho tro"

    return upload_path, None


def _image_digest(image):
    normalized = image.convert("RGB").resize((224, 224))
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _extract_fused_vector(image):
    return build_combined_vector(
        image,
        model=model,
        transform=transform,
        device=device,
    )


def _predict_category(image):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier_model(img_t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[top_idx], float(probs[top_idx].item())

def _similarity_from_ip(inner_product, min_ip, max_ip):
    span = max(max_ip - min_ip, 1e-6)
    return float((inner_product - min_ip) / span)


@lru_cache(maxsize=512)
def _product_color_hog(image_path):
    full_path = os.path.join("datasets", image_path.replace("/", os.sep))
    with Image.open(full_path).convert("RGB") as image:
        return extract_color_histogram(image), extract_hog_features(image)


def _style_key(name, category):
    text = str(name or "").lower()
    category_text = str(category or "").lower()
    text = re.sub(rf"\b{re.escape(category_text)}s?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _predict_style(category_candidates):
    style_votes = {}
    for similarity, _, info in category_candidates[:24]:
        style = _style_key(info["name"], info["category"])
        if not style:
            continue
        weight = max(float(similarity), 1e-6)
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
def _product_digest(image_path):
    full_path = os.path.join("datasets", image_path.replace("/", os.sep))
    with Image.open(full_path).convert("RGB") as image:
        return _image_digest(image)


def _absolute_cosine_score(cosine_sim):
    if cosine_sim >= 0.92:
        return 1.0
    if cosine_sim >= 0.82:
        return 0.9
    if cosine_sim >= 0.72:
        return 0.72
    if cosine_sim >= 0.62:
        return 0.5
    if cosine_sim >= 0.52:
        return 0.28
    return 0.16


def _is_out_of_domain(max_similarity, category_confidence):
    return category_confidence < 0.45 and max_similarity < 0.42


def _rank_similar_products(image, top_k=9):
    query_vec = _extract_fused_vector(image)
    query_color_hist = extract_color_histogram(image)
    query_hog_hist = extract_hog_features(image)
    query_digest = _image_digest(image)
    predicted_category, category_confidence = _predict_category(image)

    search_k = index.ntotal
    similarities, indices = index.search(query_vec.reshape(1, -1), k=search_k)

    candidates = [
        (float(sim), int(idx))
        for sim, idx in zip(similarities[0], indices[0])
        if idx >= 0
    ]
    if not candidates:
        return [], None

    best_similarity = candidates[0][0]
    out_of_domain = _is_out_of_domain(best_similarity, category_confidence)

    category_candidates = []
    fallback_candidates = []
    for similarity, idx in candidates:
        info = get_product_info(image_names[idx])
        if not info:
            continue
        row = (similarity, idx, info)
        if info["category"] == predicted_category:
            category_candidates.append(row)
        else:
            fallback_candidates.append(row)

    ranked_candidates = [] if out_of_domain else category_candidates
    if len(ranked_candidates) < top_k:
        ranked_candidates = category_candidates + fallback_candidates
    predicted_style = _predict_style(category_candidates)

    valid_similarities = [sim for sim, _, _ in ranked_candidates]
    min_sim = min(valid_similarities) if valid_similarities else 0.0
    max_sim = max(valid_similarities) if valid_similarities else 1.0
    scored = []
    seen = set()

    for similarity, idx, info in ranked_candidates:
        if not info or info["id"] in seen:
            continue
        seen.add(info["id"])

        product_color_hist, product_hog_hist = _product_color_hog(info["image_path"])
        exact_image_match = query_digest == _product_digest(info["image_path"])

        vector_score = _similarity_from_ip(similarity, min_sim, max_sim)
        absolute_score = _absolute_cosine_score(similarity)

        category_score = 1.0 if info["category"] == predicted_category else 0.0

        color_score = cosine_similarity(query_color_hist, product_color_hist)
        texture_score = cosine_similarity(query_hog_hist, product_hog_hist)
        texture_penalty = 1.0 if texture_score >= 0.55 else 0.5
        product_style = _style_key(info["name"], info["category"])
        style_match = bool(predicted_style and product_style == predicted_style)
        priority, shape_style_score = _ranking_priority(
            style_match,
            color_score,
            vector_score,
            texture_score,
        )

        if exact_image_match and category_score == 1.0:
            final_score = 0.99
        else:
            base_score = (
                0.40 * shape_style_score
                + 0.35 * color_score
                + 0.20 * texture_score
                + 0.05 * category_score
            ) * texture_penalty * (0.65 + 0.35 * absolute_score)

            if color_score < 0.45 and texture_score < 0.45:
                base_score *= 0.76

            priority_bands = {
                3: (0.72, 0.16),
                2: (0.56, 0.13),
                1: (0.42, 0.11),
                0: (0.24, 0.13),
            }
            band_min, band_width = priority_bands[priority]
            final_score = band_min + (band_width * base_score)

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
        item["color"] = item.get("color") or item.get("primary_color")
        item["color_label"] = item.get("color_label")
        item["match"] = f"{int(round(max(1, min(99, final_score * 100))))}%"
        scored.append(item)
    
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
    all_products = get_all_products() # Giáº£ sá»­ hÃ m nÃ y tráº£ vá» list toÃ n bá»™ sp
    
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
            "is_admin": _is_admin_logged_in(request),
        },
    )


@app.get("/search-image", response_class=HTMLResponse)
async def customer_search_page(request: Request):
    return templates.TemplateResponse(
        request,
        "customer_search.html",
        {"is_admin": _is_admin_logged_in(request)},
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


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request):
    if not _is_admin_logged_in(request):
        return _admin_login_redirect()
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        {"active_admin_nav": ""},
    )


@app.get("/admin/inventory", response_class=HTMLResponse)
async def admin_inventory_page(request: Request):
    if not _is_admin_logged_in(request):
        return _admin_login_redirect()
    return templates.TemplateResponse(
        request,
        "admin_inventory.html",
        {
            "orders": ORDERS,
            "inventory": _inventory_rows(),
            "active_admin_nav": "inventory",
        },
    )


@app.get("/admin/procurement", response_class=HTMLResponse)
async def admin_procurement_page(request: Request):
    if not _is_admin_logged_in(request):
        return _admin_login_redirect()
    return templates.TemplateResponse(
        request,
        "admin_procurement.html",
        {"active_admin_nav": "procurement"},
    )


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    if _is_admin_logged_in(request):
        return _admin_portal_redirect()
    return templates.TemplateResponse(
        request,
        "admin_login.html",
        {"error": None, "username": "admin"},
    )


@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if _verify_admin_credentials(username, password):
        response = RedirectResponse(url="/admin", status_code=303)
        return _set_admin_session(response)

    return templates.TemplateResponse(
        request,
        "admin_login.html",
        {
            "error": "Invalid admin username or password.",
            "username": username,
        },
        status_code=401,
    )


@app.get("/admin/register", response_class=HTMLResponse)
async def admin_register_page(request: Request):
    if _is_admin_logged_in(request):
        return _admin_portal_redirect()
    return templates.TemplateResponse(
        request,
        "admin_register.html",
        {"error": None, "username": ""},
    )


@app.post("/admin/register", response_class=HTMLResponse)
async def admin_register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = str(username).strip()
    if len(username) < 3:
        return templates.TemplateResponse(
            request,
            "admin_register.html",
            {"error": "Username must be at least 3 characters.", "username": username},
            status_code=400,
        )
    if password != confirm_password:
        return templates.TemplateResponse(
            request,
            "admin_register.html",
            {"error": "Passwords do not match.", "username": username},
            status_code=400,
        )
    if len(password) < 6:
        return templates.TemplateResponse(
            request,
            "admin_register.html",
            {"error": "Password must be at least 6 characters.", "username": username},
            status_code=400,
        )
    if username in ADMIN_ACCOUNTS:
        return templates.TemplateResponse(
            request,
            "admin_register.html",
            {"error": "Username already exists.", "username": username},
            status_code=409,
        )

    ADMIN_ACCOUNTS[username] = password
    response = RedirectResponse(url="/admin", status_code=303)
    return _set_admin_session(response)


@app.get("/admin/logout")
async def admin_logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@app.post("/api/search-image")
async def customer_search_image_api(file: UploadFile = File(...)):
    upload_path, error = _save_validated_upload(file)
    if error:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": error, "results": []},
        )

    image = Image.open(upload_path).convert("RGB")
    results, predicted_category = _rank_similar_products(image, top_k=9)
    results = _enrich_customer_search_results(results)

    if not results:
        return JSONResponse(
            content=jsonable_encoder(
                {
                    "status": "no-results",
                    "message": "No matching products found.",
                    "results": [],
                }
            )
        )

    low_confidence = all(
        float(item.get("category_confidence", 0)) < 45 for item in results
    )
    if low_confidence:
        return JSONResponse(
            content=jsonable_encoder(
                {
                    "status": "no-fashion",
                    "message": "We cannot find fashion products matching this image.",
                    "results": [],
                }
            )
        )

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "success",
                "message": f"Found {len(results)} matching products"
                + (f" in {predicted_category}" if predicted_category else ""),
                "detected_category": predicted_category,
                "results": results,
            }
        )
    )


@app.post("/api/orders")
async def create_order(payload: OrderCreateRequest):
    _ensure_inventory()
    product_id = str(payload.product_id)
    quantity = max(1, int(payload.quantity or 1))
    inventory_item = INVENTORY_LEDGER.get(product_id)
    product = _get_product_by_id(product_id)

    if not product or not inventory_item:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Product not found"},
        )
    if inventory_item["atp"] < quantity:
        return JSONResponse(
            status_code=409,
            content={"status": "error", "message": "Insufficient ATP"},
        )

    inventory_item["atp"] -= quantity
    inventory_item["reserved"] += quantity
    inventory_item["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    order = {
        "order_id": f"SO-{len(ORDERS) + 1:04d}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sku": product_id,
        "product_name": product["name"],
        "category": product["category"],
        "quantity": quantity,
        "price": product["price"],
        "status": "Reserved",
        "remaining_atp": inventory_item["atp"],
    }
    ORDERS.insert(0, order)

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "success",
                "message": f"Created {order['order_id']} and reserved stock",
                "order": order,
                "inventory": inventory_item,
            }
        )
    )


@app.get("/api/orders")
async def get_orders_api():
    return JSONResponse(content=jsonable_encoder({"orders": ORDERS}))


@app.get("/api/inventory")
async def get_inventory_api():
    return JSONResponse(content=jsonable_encoder({"inventory": _inventory_rows()}))


@app.post("/api/procurement/search-image")
async def procurement_search_image_api(request: Request, file: UploadFile = File(...)):
    if not _is_admin_logged_in(request):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Admin login required", "results": []},
        )
    await asyncio.sleep(1.1)
    upload_path, error = _save_validated_upload(file)
    if error:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": error, "results": []},
        )

    image = Image.open(upload_path).convert("RGB")
    results, predicted_category = _rank_similar_products(image, top_k=10)
    results = [_with_inventory(item) for item in results]

    if not results:
        return JSONResponse(
            content={
                "status": "no-results",
                "message": "No supplier matches found.",
                "results": [],
            }
        )

    for index_num, item in enumerate(results):
        item["supplier"] = f"Lumina Supplier {index_num + 1}"
        item["rank"] = index_num + 1
        if index_num == 0:
            item["atp"] = 0
            item["availability_label"] = "Out of Stock"
            item["orderable"] = False
            item["suggested_alternative"] = False
        else:
            item["availability_label"] = f"ATP: {item['atp']}"
            item["orderable"] = True
            item["suggested_alternative"] = True

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "success",
                "message": "Supplier matches returned with exception handling.",
                "detected_category": predicted_category,
                "results": results,
            }
        )
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
