"""
Combined image features: ResNet50 (2048) + Color Histogram + LBP texture.
Vectors are L2-normalized per block, weighted, then fused for cosine search.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Block weights after per-block L2 normalization (sum ~= 1.0)
RESNET_WEIGHT = 0.30
COLOR_WEIGHT = 0.35
HOG_WEIGHT = 0.35

COLOR_BINS_PER_CHANNEL = 16
HOG_SIZE = (128, 128)
COLOR_SIZE = (128, 128)

RESNET_DIM = 2048
COLOR_DIM = COLOR_BINS_PER_CHANNEL * 3
HOG_DIM = 1764
FUSED_DIM = RESNET_DIM + COLOR_DIM + HOG_DIM


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return (vec / norm).astype(np.float32)


def extract_color_histogram(image: Image.Image) -> np.ndarray:
    """RGB histogram (16 bins per channel) -> 48 dims, L2-normalized."""
    sample = image.convert("RGB").resize(COLOR_SIZE)
    pixels = np.asarray(sample, dtype=np.uint8)
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(
            pixels[:, :, channel],
            bins=COLOR_BINS_PER_CHANNEL,
            range=(0, 256),
        )
        parts.append(hist.astype(np.float32))
    hist = np.concatenate(parts)
    return l2_normalize(hist)


def extract_hog_features(image: Image.Image) -> np.ndarray:
    """HOG features for fabric shape/texture using cv2."""
    import cv2
    gray = np.asarray(image.convert("L").resize(HOG_SIZE), dtype=np.uint8)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9
    )
    hist = hog.compute(gray).flatten()
    return l2_normalize(hist)


def extract_resnet_vector(image: Image.Image, model, transform, device=None) -> np.ndarray:
    import torch

    img_t = transform(image).unsqueeze(0)
    if device is not None:
        img_t = img_t.to(device)
    with torch.no_grad():
        features = model(img_t).flatten().cpu().numpy().astype(np.float32)
    return l2_normalize(features)


def build_combined_vector(
    image: Image.Image,
    resnet_vector: np.ndarray | None = None,
    *,
    model=None,
    transform=None,
    device=None,
) -> np.ndarray:
    """
    Fuse ResNet50 + color histogram + HOG into one L2-normalized vector
    for FAISS IndexFlatIP (cosine similarity).
    """
    if resnet_vector is None:
        if model is None or transform is None:
            raise ValueError("model and transform are required when resnet_vector is None")
        resnet_vector = extract_resnet_vector(image, model, transform, device)

    resnet_part = l2_normalize(resnet_vector) * RESNET_WEIGHT
    color_part = l2_normalize(extract_color_histogram(image)) * COLOR_WEIGHT
    hog_part = l2_normalize(extract_hog_features(image)) * HOG_WEIGHT
    fused = np.concatenate([resnet_part, color_part, hog_part]).astype(np.float32)
    return l2_normalize(fused)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors (= dot product)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
