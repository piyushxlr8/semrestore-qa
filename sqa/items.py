"""One table of everything that gets scored: (reference, test image) pairs with labels.

kind             what the test image is                                     label
---------------  ---------------------------------------------------------  ----------
edit             reference with one controlled content change              positive
control_blur     reference blurred to the SAME PSNR as one edit            negative
control_noise    reference + noise to the SAME PSNR as one edit            negative
degraded         reference blurred/noised on a fixed grid (calib + SBC)    negative
restored_ref     SR output of the clean reference (real restorer)          negative*
restored_edit    SR output of an edited image, compared to the CLEAN ref   positive*

* "edit-then-restore": the restorer sees an image whose content differs from the reference,
  exactly like a restorer that hallucinated. Same restorer artifacts on both sides, so the only
  difference between restored_edit and restored_ref is the semantic change.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["item_id", "kind", "image_id", "split", "ref_path", "test_path", "edit_id", "edit_type",
           "severity", "mask_path", "family", "param"]


def _read(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def build_items(data_dir) -> pd.DataFrame:
    d = Path(data_dir)
    images = pd.read_csv(d / "images.csv").set_index("image_id")
    edits = _read(d / "edits" / "edits.csv")
    controls = _read(d / "controls" / "controls.csv")
    degraded = _read(d / "degraded" / "degraded.csv")
    restored = _read(d / "restored" / "restored.csv")
    rows = []

    if len(edits):
        for e in edits[edits.valid == 1].itertuples():
            rows.append(dict(item_id=e.edit_id, kind="edit", image_id=e.image_id, split=e.split,
                             ref_path=images.loc[e.image_id, "ref_path"], test_path=e.edited_path,
                             edit_id=e.edit_id, edit_type=e.edit_type, severity=e.severity,
                             mask_path=e.mask_path))
    if len(controls):
        ed = edits.set_index("edit_id")
        for c in controls.itertuples():
            e = ed.loc[c.edit_id]
            rows.append(dict(item_id=c.control_id, kind=f"control_{c.control_type}", image_id=e.image_id,
                             split=e.split, ref_path=images.loc[e.image_id, "ref_path"],
                             test_path=c.control_path, edit_id=c.edit_id, edit_type=e.edit_type,
                             param=c.param))
    if len(degraded):
        for g in degraded.itertuples():
            rows.append(dict(item_id=g.degraded_id, kind="degraded", image_id=g.image_id, split=g.split,
                             ref_path=images.loc[g.image_id, "ref_path"], test_path=g.path,
                             edit_type=g.degradation, param=g.param))
    if len(restored):
        ed = edits.set_index("edit_id") if len(edits) else None
        for r in restored.itertuples():
            is_edit = isinstance(r.source, str) and r.source != "ref"
            row = dict(item_id=r.restore_id, kind="restored_edit" if is_edit else "restored_ref",
                       image_id=r.image_id, split=r.split, ref_path=images.loc[r.image_id, "ref_path"],
                       test_path=r.out_path, family=r.family)
            if is_edit:
                e = ed.loc[r.source]
                row.update(edit_id=r.source, edit_type=e.edit_type, severity=e.severity, mask_path=e.mask_path)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
