"""Step 04: real super-resolution models from three families (x4, 128 -> 512).

  regression  Swin2SR        (trained with pixel loss: tends to be blurry but faithful)
  gan         Real-ESRGAN    (adversarial loss: sharp, can invent texture)
  diffusion   SD x4 upscaler (generative prior: sharpest, most likely to hallucinate)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import bicubic_resize, to_uint8


def degrade_sr_x4(img: np.ndarray) -> np.ndarray:
    return bicubic_resize(img, img.shape[0] // 4)


class Swin2SRRestorer:
    family = "regression"

    def __init__(self, model_id, device):
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
        self.proc = Swin2SRImageProcessor.from_pretrained(model_id)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device).eval()
        self.device = device

    def __call__(self, lr, seed=0):
        import torch
        h, w = lr.shape[:2]
        inputs = self.proc(Image.fromarray(to_uint8(lr)), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs).reconstruction[0]
        out = out.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return out[: 4 * h, : 4 * w].astype(np.float32)   # remove the processor's padding


class RealESRGANRestorer:
    family = "gan"

    def __init__(self, weights_url, device):
        from pathlib import Path
        from spandrel import ModelLoader
        from torch.hub import download_url_to_file, get_dir
        path = Path(get_dir()) / "checkpoints" / Path(weights_url).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            download_url_to_file(weights_url, str(path))
        self.model = ModelLoader().load_from_file(str(path)).to(device).eval()
        self.device = device

    def __call__(self, lr, seed=0):
        import torch
        x = torch.from_numpy(lr.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.model(x)[0]
        return out.clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)


class SDUpscaleRestorer:
    family = "diffusion"

    def __init__(self, model_id, steps, noise_level, guidance_scale, prompt, device):
        import torch
        from diffusers import StableDiffusionUpscalePipeline
        half = device in ("cuda", "mps")
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if half else torch.float32,
            variant="fp16" if half else None).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if device != "cpu":
            self.pipe.enable_attention_slicing()
        self.kw = dict(num_inference_steps=steps, noise_level=noise_level, guidance_scale=guidance_scale)
        self.prompt, self.device = prompt, device

    def __call__(self, lr, seed=0):
        import torch
        g = torch.Generator("cpu").manual_seed(int(seed))
        out = self.pipe(prompt=self.prompt, image=Image.fromarray(to_uint8(lr)), generator=g, **self.kw).images[0]
        size = lr.shape[0] * 4
        return np.asarray(out.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0


class FakeRestorer:
    """Bicubic upsampling plus a family-specific tweak. Only used by tests and the smoke run."""

    def __init__(self, family):
        self.family = family

    def __call__(self, lr, seed=0):
        up = bicubic_resize(lr, lr.shape[0] * 4)
        if self.family == "regression":
            return up
        rng = np.random.default_rng(seed)
        amount = 0.02 if self.family == "gan" else 0.05
        return np.clip(up + amount * rng.standard_normal(up.shape), 0, 1).astype(np.float32)


def load_restorer(family: str, cfg: dict, device: str, fake: bool = False):
    if fake:
        return FakeRestorer(family)
    fc = cfg["restore"]["families"][family]
    if family == "regression":
        return Swin2SRRestorer(fc["model"], device)
    if family == "gan":
        return RealESRGANRestorer(fc["weights_url"], device)
    if family == "diffusion":
        return SDUpscaleRestorer(fc["model"], fc["steps"], fc["noise_level"], fc["guidance_scale"],
                                 fc["prompt"], device)
    raise ValueError(family)
