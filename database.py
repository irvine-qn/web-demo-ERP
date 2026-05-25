import os

import pandas as pd

PRODUCTS_CSV = "datasets/products.csv"

_PRODUCTS_DF = None


def _load_products_df():
    """Load and normalize products.csv (supports current and legacy column names)."""
    global _PRODUCTS_DF
    if _PRODUCTS_DF is not None:
        return _PRODUCTS_DF

    df = pd.read_csv(PRODUCTS_CSV)
    df.columns = df.columns.str.strip()

    if "Link" in df.columns:
        rename_map = {
            "ID": "id",
            "Type": "type",
            "Name": "name",
            "Price": "price",
            "Link": "image_path",
            "Primary Color": "primary_color",
            "Color Label": "color_label",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    elif "ID" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"ID": "id"})

    if "type" not in df.columns and "category" in df.columns:
        df["type"] = df["category"]

    required = {"id", "name", "price", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"products.csv missing columns: {sorted(missing)}")

    _PRODUCTS_DF = df
    return _PRODUCTS_DF


def invalidate_products_cache():
    """Clear cached CSV (use after external edits while server is running)."""
    global _PRODUCTS_DF
    _PRODUCTS_DF = None


def _format_product(row):
    image_path = str(row["image_path"]).replace("\\", "/")
    category = str(row.get("type") or row.get("category", "")).strip()
    primary_color = str(row.get("primary_color", "") or "").strip()
    color_label = str(row.get("color_label", "") or "").strip()

    return {
        "id": str(row["id"]).strip(),
        "name": str(row["name"]).strip(),
        "category": category,
        "price": row["price"],
        "image_path": image_path,
        "image": f"/datasets/{image_path}",
        "primary_color": primary_color,
        "color_label": color_label,
        "color": primary_color,
    }


def get_all_products():
    """Return every product from products.csv for the store grid."""
    try:
        df = _load_products_df()
        return [_format_product(row) for _, row in df.iterrows()]
    except Exception as e:
        print(f"Loi doc CSV: {e}")
        return []


def get_categories():
    """Return product types/categories in CSV order."""
    try:
        df = _load_products_df()
        category_col = "type" if "type" in df.columns else "category"
        return list(dict.fromkeys(df[category_col].dropna().astype(str).str.strip()))
    except Exception as e:
        print(f"Loi doc category tu CSV: {e}")
        return []


def get_product_by_id(product_id):
    try:
        df = _load_products_df()
        product_id = str(product_id).strip()
        match = df[df["id"].astype(str).str.strip() == product_id]
        if not match.empty:
            return _format_product(match.iloc[0])
    except Exception as e:
        print(f"Loi doc CSV theo ID: {e}")
    return None


def get_product_info(image_name):
    """
    Return product information by image file name or path in Link/image_path column.
    """
    try:
        df = _load_products_df()
        if isinstance(image_name, bytes):
            image_name = image_name.decode("utf-8")
        image_name = str(image_name).replace("\\", "/").strip()
        basename = os.path.basename(image_name)

        paths = df["image_path"].astype(str).str.replace("\\", "/")
        match = df[
            paths.str.endswith(image_name, na=False)
            | paths.str.endswith(basename, na=False)
            | (paths == image_name)
        ]
        if not match.empty:
            return _format_product(match.iloc[0])
    except Exception as e:
        print(f"Loi doc CSV: {e}")
    return None
