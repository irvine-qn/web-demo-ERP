import pandas as pd
from database import SessionLocal, Product, Base, engine

def load_csv_to_db(csv_path="datasets/products.csv"):
    """Load CSV dataset vào SQLite database"""

    # Đọc CSV
    df = pd.read_csv(csv_path)

    # Tạo session
    db = SessionLocal()

    try:
        # Xóa dữ liệu cũ (tuỳ chọn)
        db.query(Product).delete()
        db.commit()
        print("✓ Xóa dữ liệu cũ")

        # Thêm sản phẩm từ CSV
        for _, row in df.iterrows():
            product = Product(
                name=row['name'],
                category=row['category'],
                description=row['description'],
                price=float(row['price']),
                image_url=row['image_url'],
                color=row.get('color', ''),
                form=row.get('form', '')
            )
            db.add(product)

        db.commit()
        print(f"✓ Tải {len(df)} sản phẩm từ CSV vào database")

    except Exception as e:
        print(f"✗ Lỗi: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    load_csv_to_db()
