# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./demo.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
#id,name,category,price,image_url
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)       # Tên quần/áo
    category = Column(String)               # Danh mục (Áo thun, Váy, Quần jean...)
    price = Column(Float)                   # Giá tiền
    image_url = Column(String)              # Đường dẫn ảnh mẫu

Base.metadata.create_all(bind=engine)