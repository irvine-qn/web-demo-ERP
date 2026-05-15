import torch
import torch.nn as nn
from torchvision import models, transforms


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _build_resnet50(pth_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    return model


def get_model(pth_path):
    model = _build_resnet50(pth_path)
    feature_model = nn.Sequential(*list(model.children())[:-1])
    feature_model.eval()
    return feature_model


def get_classifier_model(pth_path):
    return _build_resnet50(pth_path)
