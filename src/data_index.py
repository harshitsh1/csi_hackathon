"""Capture discovery + development/validation split.

Room A (Experiment-2): the supplied `validation/` folder is the validation set; everything else is dev.
Room B (Experiment-3): 25 captures per activity are drawn at capture level with seed 42 from the
sorted file names (per class, fresh random.Random(seed) each); the rest is development.
If Experiment-3 already contains `development/` / `validation/` folders they are honoured.
Raw files are never moved or modified; the split is recorded in a manifest.
"""
import random
from dataclasses import dataclass
from pathlib import Path

from .labels import label_room_a, label_room_b


@dataclass
class CaptureRecord:
    capture_id: str
    room: str
    split: str          # development | validation
    label: str
    path: str


def _skip(p: Path):
    return any(x.startswith(".") or x == "__MACOSX" for x in p.parts)


def index_room_a(root):
    root = Path(root)
    base = root / "Experiment-2" if (root / "Experiment-2").exists() else root
    recs = []
    for p in sorted(base.rglob("*.pcap")):
        rel = p.relative_to(base)
        if _skip(rel):
            continue
        lab = label_room_a(p)
        if lab is None:
            continue
        split = "validation" if "validation" in [x.lower() for x in rel.parts[:-1]] else "development"
        recs.append(CaptureRecord(f"A_{split[:3]}_{lab.lower()}_{p.stem}", "A", split, lab, str(p)))
    return recs


def index_room_b(root, val_per_class=25, seed=42):
    root = Path(root)
    base = root / "Experiment-3" if (root / "Experiment-3").exists() else root
    files = []
    for p in sorted(base.rglob("*.csv")):
        rel = p.relative_to(base)
        if _skip(rel) or "intermediate" in str(rel).lower():
            continue
        lab = label_room_b(p)
        if lab is not None:
            files.append((p, lab, [x.lower() for x in rel.parts[:-1]]))
    pre = any(("validation" in parts) for _, _, parts in files)
    recs = []
    if pre:
        for p, lab, parts in files:
            split = "validation" if "validation" in parts else "development"
            recs.append(CaptureRecord(f"B_{split[:3]}_{lab.lower()}_{p.stem}", "B", split, lab, str(p)))
        return recs
    by_label = {}
    for p, lab, _ in files:
        by_label.setdefault(lab, []).append(p)
    for lab in sorted(by_label):
        names = sorted(by_label[lab], key=lambda q: q.name)
        if len(names) <= val_per_class:
            raise ValueError(f"Room B class {lab}: only {len(names)} captures, cannot hold out {val_per_class}. "
                             f"Lower split.room_b_val_per_class for toy data.")
        val = set(random.Random(seed).sample([q.name for q in names], val_per_class))
        for q in names:
            split = "validation" if q.name in val else "development"
            recs.append(CaptureRecord(f"B_{split[:3]}_{lab.lower()}_{q.stem}", "B", split, lab, str(q)))
    return recs


def index_all(cfg):
    root = cfg["paths"]["data_root"]
    recs = index_room_a(root) + index_room_b(root, cfg["split"]["room_b_val_per_class"],
                                             cfg["split"]["room_b_seed"])
    ids = [r.capture_id for r in recs]
    assert len(ids) == len(set(ids)), "duplicate capture ids"
    return recs
