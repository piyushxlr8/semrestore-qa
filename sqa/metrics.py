"""All metrics in one place.

Part A (needs a GPU, run once in step 05): for every (reference, test) pair compute
  * classic scores: PSNR, SSIM, MS-SSIM, LPIPS, DISTS, DINO global (CLS) distance
  * SHAFE-ResNet: faithful re-implementation of the official HalluGen SHAFE.py
  * DINOv2 per-layer patch distance maps (L x 32 x 32), saved to disk

Part B (pure numpy, instant): turn saved DINOv2 maps into scores. This is where our metric
lives, and why tuning on dev is cheap: we never re-run the network.

    SHAFE-DINO      mean over layers of patch distance -> softmax pooling   (= SHAFE recipe)
    + calibration   divide each layer's distances by what blur/noise alone causes on dev images
    + top-k pooling average of the k% worst patches
    SemRestore      calibration + top-k pooling   (our metric)

Convention: every metric is turned into BADNESS (higher = more likely the content changed),
so all evaluation code treats every metric the same way.
"""
from __future__ import annotations

import numpy as np

from .utils import psnr

# name -> True if a HIGHER raw value means a BETTER (more faithful) image
HIGHER_IS_BETTER = {
    "psnr": True, "ssim": True, "ms_ssim": True,
    "lpips": False, "dists": False, "dino_global": False, "shafe_resnet": False,
    "shafe_dino": False, "shafe_dino_calib": False, "semrestore_topk": False, "semrestore": False,
}

# encoder "family" of every learned metric; edits/validity models must not share these
METRIC_ENCODER_FAMILY = {"lpips": "alexnet_imagenet", "dists": "vgg_imagenet",
                         "shafe_resnet": "resnet_imagenet", "dino_global": "dinov2",
                         "shafe_dino": "dinov2", "shafe_dino_calib": "dinov2",
                         "semrestore_topk": "dinov2", "semrestore": "dinov2"}
GENERATOR_FAMILIES = {"stable_diffusion", "lama", "openai_clip"}

DINO_VARIANTS = {   # name -> (use calibration?, pooling, add colour map?)
    "shafe_dino": (False, "softmax", False),
    "shafe_dino_calib": (True, "softmax", False),
    "semrestore": (True, "softmax", True),
    "semrestore_topk": (True, "topk", True),
}


def to_badness(metric: str, value):
    return -np.asarray(value, float) if HIGHER_IS_BETTER[metric] else np.asarray(value, float)


# ============================================================================ Part A: networks
def lowpass_fft(x, radius: float):
    """Circular low-pass filter in the Fourier domain (same as official SHAFE). x: (N,C,H,W)."""
    import torch
    _, _, h, w = x.shape
    yy = torch.arange(h, device=x.device) - h // 2
    xx = torch.arange(w, device=x.device) - w // 2
    mask = ((yy[:, None] ** 2 + xx[None, :] ** 2).float().sqrt() <= radius).float()
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)) * mask
    return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-2, -1))).real


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RawScorer:
    def __init__(self, cfg: dict, device: str):
        import lpips
        import piq
        import timm
        import torch
        from transformers import Dinov2Config, Dinov2Model

        self.cfg, self.device = cfg, device
        mcfg = cfg["metrics"]
        fake = bool(mcfg.get("fake_models", False))
        self.piq, self.torch = piq, torch
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        # LPIPS (AlexNet). In fake mode: random backbone, no downloads.
        self.lpips = (lpips.LPIPS(net="alex", pretrained=False, pnet_rand=True, verbose=False) if fake
                      else lpips.LPIPS(net="alex", verbose=False)).to(device).eval()
        self.dists = None if fake else piq.DISTS(reduction="none").to(device)

        s = mcfg["shafe_resnet"]
        self.shafe_cfg = s
        self.resnet = timm.create_model(s["model"], pretrained=not fake, features_only=True,
                                        out_indices=tuple(s["out_indices"])).to(device).eval()

        d = mcfg["dino"]
        self.dino_cfg = d
        if fake:
            conf = Dinov2Config(hidden_size=32, num_hidden_layers=max(d["layers"]), num_attention_heads=2,
                                intermediate_size=64, patch_size=14, image_size=d["input_size"])
            self.dino = Dinov2Model(conf)
        else:
            self.dino = Dinov2Model.from_pretrained(d["model"])
        self.dino = self.dino.to(device).eval()

    # ------------------------------------------------------------------ helpers
    def _tensor(self, imgs):
        arr = np.stack(imgs).transpose(0, 3, 1, 2)
        return self.torch.from_numpy(arr).float().to(self.device).clamp(0, 1)

    @staticmethod
    def _cosine_map(fa, fb):
        import torch.nn.functional as F
        return 1.0 - (F.normalize(fa, dim=1) * F.normalize(fb, dim=1)).sum(1)   # (N, h, w)

    def _shafe_resnet(self, ref, test):
        """Official SHAFE.py: clamp -> FFT low-pass -> ImageNet norm -> resize to 224 (bilinear)
        -> ResNet feats at out_indices [1, 2] -> upsample to the coarsest map, concat -> cosine
        distance per location -> softmax pooling with temperature 0.02."""
        import torch.nn.functional as F
        torch = self.torch
        r = self.shafe_cfg["lowpass_radius_at_256"] * ref.shape[-1] / 256.0
        a = (lowpass_fft(ref, r) - self.mean) / self.std
        b = (lowpass_fft(test, r) - self.mean) / self.std
        size = self.shafe_cfg.get("input_size")
        if size:
            a = F.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(size, size), mode="bilinear", align_corners=False)
        fa, fb = self.resnet(a), self.resnet(b)
        size = fa[-1].shape[-2:]
        fa = torch.cat([F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fa], 1)
        fb = torch.cat([F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fb], 1)
        dmap = self._cosine_map(fa, fb)                                  # (N, 28, 28) at the official 224 input
        flat = dmap.flatten(1)
        score = (torch.softmax(flat / self.shafe_cfg["temperature"], 1) * flat).sum(1)
        return score, dmap

    def _dino(self, ref, test):
        import torch.nn.functional as F
        d = self.dino_cfg
        if d.get("lowpass_radius_at_256"):
            r = d["lowpass_radius_at_256"] * ref.shape[-1] / 256.0
            ref, test = lowpass_fft(ref, r), lowpass_fft(test, r)
        size = d["input_size"]
        x = self.torch.cat([ref, test], 0)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        out = self.dino(pixel_values=x, output_hidden_states=True)
        n = ref.shape[0]
        g = size // 14
        maps = []
        for layer in d["layers"]:
            tok = out.hidden_states[layer][:, 1:, :]                    # drop CLS token
            tok = tok.transpose(1, 2).reshape(2 * n, -1, g, g)
            maps.append(self._cosine_map(tok[:n], tok[n:]))
        cls = out.last_hidden_state[:, 0]
        glob = 1.0 - F.cosine_similarity(cls[:n], cls[n:], dim=1)
        return glob, self.torch.stack(maps, 1)                          # (N,), (N, L, g, g)

    # ------------------------------------------------------------------ public
    def score_batch(self, refs, tests):
        torch, piq = self.torch, self.piq
        with torch.no_grad():
            a, b = self._tensor(refs), self._tensor(tests)
            res = {
                "psnr": [psnr(r, t) for r, t in zip(refs, tests)],
                "ssim": piq.ssim(a, b, data_range=1.0, reduction="none").cpu().numpy(),
                "ms_ssim": piq.multi_scale_ssim(a, b, data_range=1.0, reduction="none").cpu().numpy(),
                "lpips": self.lpips(2 * a - 1, 2 * b - 1).flatten().cpu().numpy(),
            }
            if self.dists is not None:
                res["dists"] = self.dists(a, b).flatten().cpu().numpy()
            s, smap = self._shafe_resnet(a, b)
            res["shafe_resnet"] = s.cpu().numpy()
            glob, dmaps = self._dino(a, b)
            res["dino_global"] = glob.cpu().numpy()
        rows = [{k: float(v[i]) for k, v in res.items()} for i in range(len(refs))]
        g = self.dino_cfg["input_size"] // 14
        maps = [{"shafe_resnet": smap[i].cpu().numpy().astype(np.float16),
                 "dino": dmaps[i].cpu().numpy().astype(np.float16),
                 "color": color_map(r, t, g).astype(np.float16)} for i, (r, t) in enumerate(zip(refs, tests))]
        return rows, maps


def color_map(ref: np.ndarray, test: np.ndarray, grid: int = 32) -> np.ndarray:
    """Per-patch colour difference: average each image's Lab a*/b* (the colour, not the
    brightness) over a grid x grid set of patches, then take the distance per patch.
    Averaging over a patch is a low-pass filter, so blur and noise barely change it."""
    import cv2
    a = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 1:]
    b = cv2.cvtColor(test.astype(np.float32), cv2.COLOR_RGB2LAB)[..., 1:]
    a = cv2.resize(a, (grid, grid), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (grid, grid), interpolation=cv2.INTER_AREA)
    return np.sqrt(((a - b) ** 2).sum(-1))


# ============================================================================ Part B: pooling
def softmax_pool(m: np.ndarray, tau: float) -> float:
    flat = m.astype(np.float64).ravel()
    w = np.exp((flat - flat.max()) / tau)
    w /= w.sum()
    return float((w * flat).sum())


def topk_pool(m: np.ndarray, frac: float) -> float:
    flat = np.sort(m.astype(np.float64).ravel())[::-1]
    k = max(1, int(round(frac * flat.size)))
    return float(flat[:k].mean())


def calibration_stats(dino_maps: list[np.ndarray], color_maps: list[np.ndarray]) -> dict:
    """Mean / std of patch distances on 'blurry or noisy but correct' dev images:
    one pair per DINOv2 layer, plus one for the colour map."""
    stack = np.stack([m.astype(np.float64) for m in dino_maps])          # (n, L, g, g)
    col = np.stack([m.astype(np.float64) for m in color_maps])           # (n, g, g)
    return {"mu": stack.mean(axis=(0, 2, 3)).tolist(), "sd": (stack.std(axis=(0, 2, 3)) + 1e-6).tolist(),
            "color_mu": float(col.mean()), "color_sd": float(col.std() + 1e-6)}


def dino_heatmap(maps: np.ndarray, stored_layers, layers, calibrate: bool, stats=None,
                 color=None) -> np.ndarray:
    """Combine the selected layers of one item's (L, g, g) maps (and optionally its (g, g)
    colour map) into a single (g, g) map."""
    idx = [list(stored_layers).index(l) for l in layers]
    m = maps.astype(np.float64)[idx]
    if calibrate:
        mu = np.asarray(stats["mu"])[idx][:, None, None]
        sd = np.asarray(stats["sd"])[idx][:, None, None]
        m = np.maximum(0.0, (m - mu) / sd)       # "how many blur-stds above normal" per patch
    if color is not None:                        # colour map = one more calibrated "layer"
        zc = np.maximum(0.0, (color.astype(np.float64) - stats["color_mu"]) / stats["color_sd"])
        m = np.concatenate([m, zc[None]], 0)
    return m.mean(0)


def dino_variant_score(maps, variant, params, stored_layers, stats=None, color=None):
    calibrate, pooling, use_color = DINO_VARIANTS[variant]
    hm = dino_heatmap(maps, stored_layers, params["layers"], calibrate, stats,
                      color if use_color else None)
    score = softmax_pool(hm, params["tau"]) if pooling == "softmax" else topk_pool(hm, params["k_frac"])
    return score, hm
