import pandas as pd

PRODUCTS_CSV = "datasets/products.csv"


MASCULINE_KEYWORDS = {
    "henley",
    "oxford",
    "polo",
    "cargo",
    "chino",
    "derby",
    "loafer",
    "bomber",
}
FEMININE_KEYWORDS = {
    "dress",
    "skirt",
    "floral",
    "heels",
    "blouse",
    "gown",
    "sandal",
}


def classify_gender_style(name, category):
    """Classify fashion style into masculine, feminine, or neutral."""
    text = f"{name or ''} {category or ''}".lower()
    if category == "Dress" or any(keyword in text for keyword in FEMININE_KEYWORDS):
        return "Feminine"
    if any(keyword in text for keyword in MASCULINE_KEYWORDS):
        return "Masculine"
    return "Neutral"


def _format_product(row):
    image_path = str(row["image_path"]).replace("\\", "/")
    name = row["name"]
    category = row["category"]
    return {
        "id": row["id"],
        "name": name,
        "category": category,
        "gender_style": classify_gender_style(name, category),
        "price": row["price"],
        "image_path": image_path,
        "image": f"/datasets/{image_path}",
    }


def get_all_products():
    """Return every product from products.csv for the store grid."""
    try:
        df = pd.read_csv(PRODUCTS_CSV)
        return [_format_product(row) for _, row in df.iterrows()]
    except Exception as e:
        print(f"Loi doc CSV: {e}")
        return []


def get_categories():
    """Return database categories in CSV order."""
    try:
        df = pd.read_csv(PRODUCTS_CSV)
        return list(dict.fromkeys(df["category"].dropna().astype(str)))
    except Exception as e:
        print(f"Loi doc category tu CSV: {e}")
        return []


def get_product_info(image_name):
    """
    Return product information by image file name.
    CSV columns: id, name, category, price, image_path.
    """
    try:
        df = pd.read_csv(PRODUCTS_CSV)
        if isinstance(image_name, bytes):
            image_name = image_name.decode("utf-8")
        image_name = str(image_name).replace("\\", "/")
        product = df[df["image_path"].str.endswith(image_name, na=False)]
        if not product.empty:
            return _format_product(product.iloc[0])
    except Exception as e:
        print(f"Loi doc CSV: {e}")
    return None
