import torch
import torch.nn as nn
from torchvision import models, transforms

# Tiền xử lý ảnh chuẩn ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(pth_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6) # 6 lớp như đã train
    
    model.load_state_dict(torch.load(pth_path, map_location=device))
    # Cắt bỏ lớp cuối để lấy đặc trưng 2048 chiều
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model