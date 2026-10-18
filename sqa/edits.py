"""Step 02: controlled semantic edits with a known mask, type and severity.

Four edit types (all are "content changes" a good metric must catch):
  addition      a new object appears where there was background   (= classic hallucination)
  removal       an existing object disappears                      (LaMa inpainting)
  substitution  an object is replaced by a different category      (Stable Diffusion inpainting)
  recolor       the object keeps its shape but changes colour      (HSV hue shift, no model)

Every edit is pasted back onto the reference, so pixels outside the mask are untouched.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .utils import bbox_of, to_uint8


# ----------------------------------------------------------------------------- models
class SDInpainter:
    family = "stable_diffusion"

    def __init__(self, model_id: str, steps: int, guidance: float, device: str):
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        half = device in ("cuda", "mps")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if half else torch.float32,
            variant="fp16" if half else None).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if device != "cpu":
            self.pipe.enable_attention_slicing()
        self.steps, self.guidance, self.device = steps, guidance, device

    def __call__(self, img, region, prompt, negative, seed):
        import torch
        g = torch.Generator("cpu").manual_seed(int(seed))   # CPU generator: same noise on any GPU
        h, w = region.shape
        out = self.pipe(prompt=prompt, negative_prompt=negative,
                        image=Image.fromarray(to_uint8(img)),
                        mask_image=Image.fromarray(region.astype(np.uint8) * 255),
                        height=h, width=w, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=g).images[0]
        return np.asarray(out.resize((w, h)), dtype=np.float32) / 255.0


class LamaInpainter:
    """Big-LaMa (TorchScript) for clean object removal. Loaded directly so we avoid the
    `simple-lama-inpainting` package, which pins old numpy/pillow versions."""
    family = "lama"

    def __init__(self, url: str, device: str):
        import torch
        from torch.hub import download_url_to_file, get_dir
        from pathlib import Path

        path = Path(get_dir()) / "checkpoints" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            download_url_to_file(url, str(path))
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.device = device

    def __call__(self, img, region, *args, **kwargs):
        import torch
        h, w = region.shape
        H, W = int(np.ceil(h / 8) * 8), int(np.ceil(w / 8) * 8)
        x = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode="symmetric")
        m = np.pad(region.astype(np.float32), ((0, H - h), (0, W - w)), mode="symmetric")
        xt = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        mt = (torch.from_numpy(m)[None, None] > 0).float().to(self.device)
        with torch.inference_mode():
            out = self.model(xt, mt)[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(out[:h, :w], 0, 1).astype(np.float32)


class ClipChecker:
    """Zero-shot CLIP check on the edited region: did the content really change?"""
    family = "openai_clip"

    def __init__(self, model_id: str, device: str):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.device = device

    def probs(self, img, region, texts):
        import torch
        y0, y1, x0, x1 = bbox_of(region, margin=16)
        crop = Image.fromarray(to_uint8(img[y0:y1, x0:x1]))
        inputs = self.proc(text=texts, images=crop, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0]
        return logits.softmax(-1).cpu().numpy()


# ----------------------------------------------------------------------------- helpers
def pick_new_category(categories, present, rng, exclude=()):
    options = [c for c in categories if c not in present and c not in exclude]
    return options[rng.integers(len(options))]


def box_around(mask: np.ndarray, margin: int = 0) -> np.ndarray:
    """Filled bounding box of a mask. Substitution inpaints this box, so the new object is not
    forced into the old object's silhouette."""
    y0, y1, x0, x1 = bbox_of(mask, margin)
    box = np.zeros_like(mask, bool)
    box[y0:y1, x0:x1] = True
    return box


def odds_drop(p_before: float, p_after: float) -> float:
    """How many times less sure CLIP is that the object is there (odds ratio before / after)."""
    eps = 1e-4
    odds = lambda p: np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))  # noqa: E731
    return float(odds(p_before) / odds(p_after))


def find_addition_box(all_objects: np.ndarray, area_frac: float, rng, tries: int = 200,
                      max_overlap: float = 0.05):
    """Find a square in the background (overlapping existing objects by < 5%)."""
    h, w = all_objects.shape
    side = int(np.sqrt(area_frac * h * w))
    best, best_overlap = None, 1.0
    for _ in range(tries):
        y, x = rng.integers(0, h - side), rng.integers(0, w - side)
        overlap = all_objects[y:y + side, x:x + side].mean()
        if overlap < best_overlap:
            best, best_overlap = (y, x), overlap
        if overlap <= max_overlap:
            break
    if best is None or best_overlap > max_overlap:
        return None
    box = np.zeros_like(all_objects, bool)
    box[best[0]:best[0] + side, best[1]:best[1] + side] = True
    return box


def recolor(img: np.ndarray, mask: np.ndarray, hue_shift_deg: float) -> np.ndarray:
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)   # H in [0, 360)
    hsv[..., 0] = np.where(mask, (hsv[..., 0] + hue_shift_deg) % 360.0, hsv[..., 0])
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)


def mean_saturation(img, mask) -> float:
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1][mask].mean())


def mean_abs_change(a, b, region) -> float:
    return float(np.abs(a - b)[region].mean())
