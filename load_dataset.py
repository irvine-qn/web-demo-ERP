import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import faiss
import numpy as np
from model_utils import get_model, transform

# Dùng để quét dataset tạo file .index
model = get_model('models/fashion_resnet50.pth')

# Cấu hình mô hình
device = torch.device("cpu")
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load('models/fashion_model.pth', map_location=device))
model = nn.Sequential(*list(model.children())[:-1]) # Lấy đặc trưng
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def create_index():
    vectors = []
    image_names = []
    root_dir = 'datasets'

    for subdir in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, subdir)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(sub_path, file)
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        vec = model(img_t).flatten().numpy()
                    vectors.append(vec)
                    image_names.append(file) # Lưu tên file để tìm trong CSV

    index = faiss.IndexFlatL2(2048)
    index.add(np.array(vectors).astype('float32'))
    faiss.write_index(index, 'models/vector_db.index')
    np.save('models/image_names.npy', image_names)
    print("Xong!")

if __name__ == "__main__":
    create_index()