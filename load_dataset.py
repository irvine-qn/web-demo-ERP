# --- HÀM TRÍCH XUẤT MÀU SẮC (HISTOGRAM) ---
def get_color_histogram(img_path):
    try:
        image = cv2.imread(img_path)
        if image is None: return np.zeros(512)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Tính Histogram cho 3 kênh màu, mỗi kênh 8 vùng chia (8x8x8 = 512 chiều)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype('float32')
    except:
        return np.zeros(512).astype('float32')

# --- KHỞI TẠO MÔ HÌNH ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dùng hàm get_model từ model_utils để đảm bảo đồng bộ với main.py
model = get_model(MODEL_PATH).to(device)

def create_index():
    vectors = []
    image_names = []

    print(f"Bắt đầu quét dataset tại: {DATASET_DIR}...")

    # Duyệt qua các thư mục con (Dress, Hat, Pant...)
    for subdir in os.listdir(DATASET_DIR):
        sub_path = os.path.join(DATASET_DIR, subdir)
        if not os.path.isdir(sub_path):
            continue
            
        print(f"Đang xử lý nhóm: {subdir}...")
        for file in os.listdir(sub_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(sub_path, file)
            
            try:
                # 1. Trích xuất đặc trưng hình dáng (ResNet50)
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    shape_vec = model(img_t).flatten()
                    # Chuẩn hóa L2 để vector có độ dài bằng 1
                    shape_vec = F.normalize(shape_vec, p=2, dim=0).cpu().numpy()

                # 2. Trích xuất đặc trưng màu sắc (Histogram)
                color_vec = get_color_histogram(img_path)
                # Chuẩn hóa L2 cho vector màu
                color_vec = color_vec / (np.linalg.norm(color_vec) + 1e-6)

                # 3. Kết hợp (Fusion) - Style 70%, Color 30%
                combined_vec = np.hstack((shape_vec * 0.7, color_vec * 0.3)).astype('float32')
                
                # Chuẩn hóa lại toàn bộ vector sau khi gộp (Quan trọng để dùng IndexFlatIP)
                combined_vec = combined_vec / (np.linalg.norm(combined_vec) + 1e-6)
            
                vectors.append(combined_vec)
                # Lưu đường dẫn tương đối để database.py dễ truy vấn (VD: Dress/abc.jpg)
                image_names.append(f"{subdir}/{file}")
                
            except Exception as e:
                print(f"Lỗi tại file {img_path}: {e}")

    # 4. Tạo và lưu Index
    vectors = np.array(vectors).astype('float32')
    dim = vectors.shape[1] # Sẽ là 2560 (2048 + 512)
    
    # Sử dụng IndexFlatIP (Inner Product) cho vector đã chuẩn hóa = Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    
    # Lưu file
    if not os.path.exists('models'): os.makedirs('models')
    faiss.write_index(index, INDEX_PATH)
    np.save(NAMES_PATH, np.array(image_names))
    
    print(f"THÀNH CÔNG: Đã tạo danh bạ AI cho {len(vectors)} sản phẩm.")
    print(f"Kích thước vector: {dim}")

if __name__ == "__main__":
    create_index()