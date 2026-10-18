"""Small shared helpers: config, paths, image I/O, PSNR, blur/noise, masks."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


# ----------------------------------------------------------------------------- config / paths
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def data_dir(cfg) -> Path:
    return Path(cfg["paths"]["data_dir"])


def results_dir(cfg) -> Path:
    return Path(cfg["paths"]["results_dir"])


def ensure(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device() -> str:
    """cuda (Colab / NVIDIA), mps (Apple Silicon Mac) or cpu."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------- image I/O
# Convention: images are float32 numpy arrays, shape (H, W, 3), values in [0, 1].
def load_image(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(img: np.ndarray, path) -> None:
    ensure(Path(path).parent)
    Image.fromarray(to_uint8(img)).save(path)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def quantize(img: np.ndarray) -> np.ndarray:
    """What an image looks like after being saved as 8-bit PNG and loaded again."""
    return to_uint8(img).astype(np.float32) / 255.0


def load_mask(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def save_mask(mask: np.ndarray, path) -> None:
    ensure(Path(path).parent)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


# ----------------------------------------------------------------------------- degradations
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def add_noise(img: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(img.shape).astype(np.float32)   # same noise pattern for every std
    return np.clip(img + std * noise, 0.0, 1.0)


def bicubic_resize(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(to_uint8(img)).resize((size, size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------- masks
def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def paste_back(ref: np.ndarray, generated: np.ndarray, region: np.ndarray, feather_px: int) -> np.ndarray:
    """Keep `generated` only inside `region`; everything outside is copied exactly from `ref`.

    The blend weight is softened *inside* the region (feathering) but is exactly 0 outside,
    so the region mask is a valid ground truth for "where the content changed".
    """
    w = region.astype(np.float32)
    if feather_px > 0:
        k = int(2 * feather_px + 1)
        w = cv2.GaussianBlur(w, (k, k), feather_px / 2.0) * region   # zero outside region
        w = np.clip(w / max(w.max(), 1e-6), 0, 1) if w.max() > 0 else w
    w = w[..., None]
    return (w * generated + (1.0 - w) * ref).astype(np.float32)


def bbox_of(mask: np.ndarray, margin: int = 0):
    ys, xs = np.where(mask)
    h, w = mask.shape
    return (max(0, ys.min() - margin), min(h, ys.max() + 1 + margin),
            max(0, xs.min() - margin), min(w, xs.max() + 1 + margin))


# ----------------------------------------------------------------------------- hashing (freeze)
def file_hash(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def write_json(obj, path) -> None:
    ensure(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}
