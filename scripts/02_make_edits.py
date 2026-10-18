"""02 - Make controlled semantic edits (addition, removal, substitution, recolor) for every image.

Output: data/edits/edits.csv, data/edits/images/*.png, data/edits/masks/*.png,
        results/figures/edit_samples.png (look at this before going further!)

Each edit stores: type, ground-truth mask (where content changed), severity, prompt, seed,
PSNR vs reference, and a `valid` flag from an automatic CLIP check.
Re-running skips edits that already exist, so it is safe to resume after a Colab disconnect.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa import edits as E  # noqa: E402
from sqa.utils import (data_dir, dilate, ensure, get_device, load_config, load_image, load_mask,  # noqa: E402
                       paste_back, psnr, quantize, results_dir, save_image, save_mask, seed_everything)
from sqa.viz import contact_sheet  # noqa: E402

BG_TEXTS = "a photo of an empty background"


def main(cfg, limit=None, redo_invalid=None, tries=None):
    seed_everything(cfg["seed"])
    ec = cfg["edits"]
    root = data_dir(cfg)
    out_dir = ensure(root / "edits")
    csv_path = out_dir / "edits.csv"
    images = pd.read_csv(root / "images.csv")
    if limit:
        images = images.head(limit)
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=["edit_id"])
    seed_offset = 0
    if redo_invalid:    # second chance for edits CLIP rejected: new seeds (and a new box/category)
        done = done[~((done.edit_type == redo_invalid) & (done.valid == 0))]
        seed_offset = 1000
        if tries:
            ec["tries"] = tries
    done_ids = set(done.edit_id)
    rows = done.to_dict("records")

    device = get_device()
    types = ec["types"]
    sd = E.SDInpainter(ec["inpaint_model"], ec["inpaint_steps"], ec["guidance_scale"], device) \
        if {"addition", "substitution"} & set(types) else None
    # LaMa and CLIP are small: on a Mac they run on the CPU so the GPU memory is left for SD
    small = "cpu" if device == "mps" else device
    lama = E.LamaInpainter(ec["lama_url"], small) if "removal" in types else None
    clip = E.ClipChecker(ec["clip_model"], small)

    def record(edit_id, img_row, etype, edited, region, severity, **extra):
        ref = load_image(img_row.ref_path)
        edited = quantize(edited)
        ep = out_dir / "images" / f"{edit_id}.png"
        mp = out_dir / "masks" / f"{edit_id}.png"
        save_image(edited, ep)
        save_mask(region, mp)
        change = E.mean_abs_change(ref, edited, region)
        valid = extra.pop("valid", True)
        if extra.pop("check_change", True):
            valid = valid and change >= ec["min_mean_abs_change"]
        rows.append(dict(edit_id=edit_id, image_id=img_row.image_id, split=img_row.split, edit_type=etype,
                         severity=round(float(severity), 4), mask_path=str(mp), edited_path=str(ep),
                         psnr=round(psnr(edited, ref), 3), mean_abs_change=round(change, 4),
                         valid=int(valid), **extra))

    t_start = time.time()
    for n, r in enumerate(images.itertuples()):
        rng = np.random.default_rng(cfg["seed"] * 100003 + n)
        ref = load_image(r.ref_path)
        obj = load_mask(r.object_mask_path)
        allobj = load_mask(r.all_objects_mask_path)
        present = set(str(r.categories_present).split(";"))
        region = dilate(obj, ec["mask_dilate_px"])
        src = r.object_category
        seed = int(rng.integers(1_000_000)) + seed_offset

        # -- removal: object disappears (LaMa fills with background).
        #    Valid if CLIP becomes at least `removal_min_odds_drop` times less sure the object is
        #    there. (An absolute "p < 0.5" is too strict: a removed horse next to its cart still
        #    looks a bit like "a horse" to CLIP because of the context.)
        eid = f"{r.image_id}_removal"
        if "removal" in types and eid not in done_ids:
            out = paste_back(ref, lama(ref, region), region, ec["feather_px"])
            texts = [f"a photo of a {src}", BG_TEXTS]
            p0, p1 = clip.probs(ref, region, texts)[0], clip.probs(out, region, texts)[0]
            drop = E.odds_drop(p0, p1)
            record(eid, r, "removal", out, region, region.mean(), source_category=src, target_category="",
                   prompt="(LaMa)", seed=seed, generator="lama", clip_prob=round(float(p1), 3),
                   clip_prob_ref=round(float(p0), 3), valid=drop >= ec["removal_min_odds_drop"])

        # -- substitution: object becomes a different category. SD inpaints the object's bounding
        #    box (room for the new shape). Up to `tries` seeds; keep the first that CLIP accepts.
        eid = f"{r.image_id}_substitution"
        if "substitution" in types and eid not in done_ids:
            new = E.pick_new_category(ec["categories"], present, rng, exclude=[src])
            prompt = f"a photo of a {new}"
            box = E.box_around(region)
            best = None
            for k in range(ec["tries"]):
                gen = sd(ref, box, prompt, f"{src}, blurry, low quality, deformed", seed + k)
                out = paste_back(ref, gen, box, ec["feather_px"])
                p = float(clip.probs(out, box, [prompt, f"a photo of a {src}"])[0])
                if best is None or p > best[1]:
                    best = (out, p, seed + k)
                if p > 0.5:
                    break
            out, p, s_used = best
            record(eid, r, "substitution", out, box, box.mean(), source_category=src,
                   target_category=new, prompt=prompt, seed=s_used, generator="stable_diffusion",
                   clip_prob=round(p, 3), valid=p > 0.5)

        # -- addition: a new object appears in the background (hallucination)
        eid = f"{r.image_id}_addition"
        if "addition" in types and eid not in done_ids:
            box = E.find_addition_box(dilate(allobj, 8), rng.uniform(*ec["addition_area_frac"]), rng)
            if box is not None:
                new = E.pick_new_category(ec["categories"], present, rng)
                prompt = f"a photo of a {new}"
                best = None
                for k in range(ec["tries"]):
                    gen = sd(ref, box, prompt, "blurry, low quality, deformed, text", seed + k)
                    out = paste_back(ref, gen, box, ec["feather_px"])
                    p = float(clip.probs(out, box, [prompt, BG_TEXTS])[0])
                    if best is None or p > best[1]:
                        best = (out, p, seed + k)
                    if p > 0.5:
                        break
                out, p, s_used = best
                record(eid, r, "addition", out, box, box.mean(), source_category="", target_category=new,
                       prompt=prompt, seed=s_used, generator="stable_diffusion",
                       clip_prob=round(p, 3), valid=p > 0.5)

        # -- recolor: same object, different colour, three severities (no model needed)
        if "recolor" in types:
            sat = E.mean_saturation(ref, obj)
            for shift in ec["recolor_hue_shifts"]:
                eid = f"{r.image_id}_recolor{shift}"
                if eid in done_ids:
                    continue
                # validity = the object is colourful enough; the small 30° shift is the LOW
                # severity level on purpose, so no minimum-change rule here
                record(eid, r, "recolor", E.recolor(ref, obj, shift), obj, shift, source_category=src,
                       target_category=f"hue+{shift}", prompt="(hue shift)", seed=0, generator="none",
                       clip_prob=np.nan, valid=sat >= ec["recolor_min_saturation"], check_change=False)

        if n == 0:
            print(f"first image took {time.time() - t_start:.1f}s (use this to estimate total time)")
        if (n + 1) % 10 == 0 or n + 1 == len(images):
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[{n + 1}/{len(images)}] edits so far: {len(rows)}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("\nValid edits by type:\n", df.groupby("edit_type").valid.agg(["sum", "count"]))

    # contact sheet for a quick visual check
    sheet = []
    for img_id in images.image_id.head(6):
        row = [load_image(images.set_index("image_id").loc[img_id, "ref_path"])]
        for t in ["addition", "removal", "substitution", "recolor90"]:
            hit = df[df.edit_id == f"{img_id}_{t}"]
            row.append(load_image(hit.edited_path.iloc[0]) if len(hit) else None)
        sheet.append(row)
    fig_path = ensure(results_dir(cfg) / "figures") / "edit_samples.png"
    contact_sheet(sheet, ["reference", "addition", "removal", "substitution", "recolor (90°)"], fig_path)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    ap.add_argument("--redo-invalid", default=None, help="e.g. addition: retry edits CLIP rejected")
    ap.add_argument("--tries", type=int, default=None, help="SD seeds per edit for the retry")
    a = ap.parse_args()
    main(load_config(a.config), a.limit, a.redo_invalid, a.tries)
