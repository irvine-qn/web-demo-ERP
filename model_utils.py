import torch
import torch.nn as nn
from torchvision import models, transforms

def get_model(model_path):
    # Định nghĩa cấu trúc giống hệt lúc train trên Colab
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5) # Giả sử bạn có 5 loại quần áo lúc train
    
    # Load trọng số đã tải từ Colab về
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Bỏ lớp cuối để lấy đặc trưng
    feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor

# Transform ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])