"""Step 01: pick COCO images, crop them to 512x512, save object masks, split dev/test."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import ensure, save_image, save_mask


def download_annotations(url: str, out_dir: Path, split: str) -> Path:
    """Download COCO instance annotations once (about 250 MB zip) and return the json path."""
    out_dir = ensure(out_dir)
    json_path = out_dir / f"instances_{split}.json"
    if json_path.exists():
        return json_path
    zip_path = out_dir / "annotations.zip"
    if not zip_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(f"annotations/instances_{split}.json", out_dir)
    (out_dir / "annotations" / f"instances_{split}.json").rename(json_path)
    return json_path


def center_crop_box(w: int, h: int, size: int):
    """Resize shorter side to `size`, then center crop. Returns (scale, left, top)."""
    scale = size / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    return scale, (nw - size) // 2, (nh - size) // 2, nw, nh


def crop_array(arr: np.ndarray, size: int, resample) -> np.ndarray:
    h, w = arr.shape[:2]
    scale, left, top, nw, nh = center_crop_box(w, h, size)
    pil = Image.fromarray(arr).resize((nw, nh), resample)
    return np.asarray(pil)[top:top + size, left:left + size]


def build_dataset(cfg) -> pd.DataFrame:
    from pycocotools.coco import COCO

    dcfg = cfg["dataset"]
    root = Path(cfg["paths"]["data_dir"])
    size = dcfg["image_size"]
    ann_path = download_annotations(dcfg["coco_annotations_url"], root / "coco", dcfg["coco_split"])
    coco = COCO(str(ann_path))
    cat_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    rng = np.random.default_rng(cfg["seed"])
    img_ids = list(coco.getImgIds())
    rng.shuffle(img_ids)

    lo, hi = dcfg["object_area_frac"]
    rows = []
    for img_id in img_ids:
        if len(rows) >= dcfg["n_images"]:
            break
        info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        if not anns:
            continue
        # find eligible target objects after cropping
        union = np.zeros((size, size), bool)
        candidates = []
        for a in anns:
            full = coco.annToMask(a).astype(np.uint8) * 255
            cropped = crop_array(full, size, Image.NEAREST) > 127
            union |= cropped
            if a.get("iscrowd", 0):
                continue
            full_area_scaled = (full > 127).sum() * (size / min(info["width"], info["height"])) ** 2
            survived = cropped.sum() / max(full_area_scaled, 1)
            frac = cropped.sum() / (size * size)
            if lo <= frac <= hi and survived >= dcfg["min_area_survival"]:
                candidates.append((a, cropped, frac))
        if not candidates:
            continue

        try:
            with urllib.request.urlopen(info["coco_url"], timeout=30) as r:
                pil = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:  # network hiccup: skip this image
            print(f"skip {img_id}: {e}")
            continue
        img = crop_array(np.asarray(pil), size, Image.BICUBIC).astype(np.float32) / 255.0

        a, obj_mask, frac = candidates[rng.integers(len(candidates))]
        image_id = f"coco{img_id:012d}"
        ref_path = root / "refs" / f"{image_id}.png"
        obj_path = root / "masks" / f"{image_id}_object.png"
        all_path = root / "masks" / f"{image_id}_allobjects.png"
        save_image(img, ref_path)
        save_mask(obj_mask, obj_path)
        save_mask(union, all_path)
        present = sorted({cat_name[x["category_id"]] for x in anns})
        rows.append(dict(image_id=image_id, coco_id=img_id, ref_path=str(ref_path),
                         object_category=cat_name[a["category_id"]], object_mask_path=str(obj_path),
                         object_area_frac=round(float(frac), 4), all_objects_mask_path=str(all_path),
                         categories_present=";".join(present)))
        print(f"[{len(rows)}/{dcfg['n_images']}] {image_id} target={rows[-1]['object_category']}")

    df = pd.DataFrame(rows)
    order = rng.permutation(len(df))
    df["split"] = "test"
    df.loc[df.index[order[: dcfg["n_dev"]]], "split"] = "dev"
    return df
