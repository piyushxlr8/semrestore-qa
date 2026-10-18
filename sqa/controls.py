"""Step 03: "faithful but degraded" controls with exactly the same PSNR as each edit.

If an edit has PSNR 24.3 dB, we blur (or add noise to) the clean reference until it also has
PSNR 24.3 dB. Now PSNR literally cannot tell them apart, but only the edit changed the content.
A good semantic metric must say the control is closer to the reference than the edit.
"""
from __future__ import annotations

import numpy as np

from .utils import add_noise, gaussian_blur, psnr, quantize


def _bisect(make, ref, target_db, lo, hi, tol, iters=40):
    """`make(p)` returns a degraded image whose PSNR falls as p grows. Find p hitting target_db."""
    best_p, best_gap, best_img = None, np.inf, None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        img = quantize(make(mid))
        value = psnr(img, ref)
        gap = abs(value - target_db)
        if gap < best_gap:
            best_p, best_gap, best_img = mid, gap, img
        if gap <= tol:
            break
        if value > target_db:   # not degraded enough
            lo = mid
        else:
            hi = mid
    return best_p, best_img, best_gap


def match_blur(ref, target_db, sigma_range=(0.05, 25.0), tol=0.1):
    return _bisect(lambda s: gaussian_blur(ref, s), ref, target_db, *sigma_range, tol)


def match_noise(ref, target_db, std_range=(0.0005, 0.6), tol=0.1, seed=0):
    return _bisect(lambda s: add_noise(ref, s, seed), ref, target_db, *std_range, tol)
