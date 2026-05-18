import os
import faiss
import numpy as np
from PIL import Image
import torch
from model_utils import get_model, transform


DATASET_DIR = "datasets"
MODEL_PATH = "models/fashion_resnet50.pth"
INDEX_PATH = "models/vector_db.index"
NAMES_PATH = "models/product_ids.npy"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(MODEL_PATH).to(device)


def create_index():
    vectors = []
    image_names = []

    print(f"Bat dau quet dataset tai: {DATASET_DIR}...")

    for subdir in sorted(os.listdir(DATASET_DIR)):
        sub_path = os.path.join(DATASET_DIR, subdir)
        if not os.path.isdir(sub_path):
            continue

        print(f"Dang xu ly nhom: {subdir}...")
        for file in sorted(os.listdir(sub_path)):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(sub_path, file)
            try:
                image = Image.open(img_path).convert("RGB")
                img_t = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    vector = model(img_t).flatten().cpu().numpy()

                vectors.append(vector.astype("float32"))
                image_names.append(file)
            except Exception as exc:
                print(f"Loi tai file {img_path}: {exc}")

    if not vectors:
        raise RuntimeError("No valid images were found to generate the index.")

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(NAMES_PATH, np.array(image_names))

    print(f"THANH CONG: Da tao danh ba AI cho {len(vectors)} san pham.")
    print(f"Kich thuoc vector: {vectors.shape[1]}")


if __name__ == "__main__":
    create_index()
