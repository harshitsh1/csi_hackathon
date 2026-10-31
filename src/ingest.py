"""Stage 1/2: source captures -> common 256-column intermediate CSV (one per capture) -> windows.
Room A: PCAP -> Nexmon payload -> complex CSI -> amplitude. Room B: amplitude CSV schema check.
Both then share cleaning, CSV export, windowing."""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

from .amplitude import csi_to_amplitude
from .cleaning import clean_capture
from .config import preprocessing_hash, workspace_dirs
from .csi_decode import decode_pcap
from .data_index import index_all
from .windowing import make_windows

COLS = [f"subcarrier_{i:03d}" for i in range(1, 257)]


def load_capture(rec, cfg):
    """Return (amplitude (N,256) float64, source_info)."""
    S = cfg["decode"]["n_subcarriers"]
    if rec["room"] == "A":
        csi, info = decode_pcap(rec["path"], cfg["decode"])
        amp = csi_to_amplitude(csi, cfg["decode"]["fftshift"])
        info["n_decoded"] = int(len(amp))
        return amp, info
    df = pd.read_csv(rec["path"])
    amp = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if amp.shape[1] != S:
        raise ValueError(f"{rec['path']}: expected {S} columns, got {amp.shape[1]}")
    return amp, dict(n_decoded=int(len(amp)))


def _work(args):
    rec, cfg, out_dir = args
    amp, info = load_capture(rec, cfg)
    clean, rep = clean_capture(amp, cfg["qc"])
    d = Path(out_dir) / f"room_{rec['room']}" / rec["split"]
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean, columns=COLS).to_csv(d / f"{rec['capture_id']}.csv", index=False, float_format="%.5f")
    return {**rec, "source_file": Path(rec["path"]).name, **info, **rep}


def run_ingest(cfg, workers=1):
    dirs = workspace_dirs(cfg)
    dirs["inter"].mkdir(parents=True, exist_ok=True)
    recs = [r.__dict__ for r in index_all(cfg)]
    print(f"[ingest] {len(recs)} captures "
          f"(A dev/val: {sum(r['room']=='A' and r['split']=='development' for r in recs)}/"
          f"{sum(r['room']=='A' and r['split']=='validation' for r in recs)}, "
          f"B dev/val: {sum(r['room']=='B' and r['split']=='development' for r in recs)}/"
          f"{sum(r['room']=='B' and r['split']=='validation' for r in recs)})")
    t0, jobs = time.time(), [(r, cfg, dirs["inter"]) for r in recs]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_work, jobs, chunksize=8))
    else:
        rows = []
        for i, j in enumerate(jobs):
            rows.append(_work(j))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)}")
    man = pd.DataFrame(rows)
    man["preproc_hash"] = preprocessing_hash(cfg)
    man.to_csv(dirs["inter"] / "manifest.csv", index=False)
    print(f"[ingest] done in {time.time()-t0:.0f}s. QC totals: " +
          ", ".join(f"{c}={int(man[c].sum())}" for c in
                    ["n_nonfinite", "n_all_zero", "n_mostly_zero", "n_duplicate"]))
    return man


def run_windows(cfg):
    """Intermediate CSV -> window_XXXXXX.npy (150,256) float32 + provenance index."""
    dirs = workspace_dirs(cfg)
    man = pd.read_csv(dirs["inter"] / "manifest.csv")
    L, h = cfg["window"]["length"], preprocessing_hash(cfg)
    index, skipped = [], []
    for split, stride in (("development", cfg["window"]["stride_dev"]),
                          ("validation", cfg["window"]["stride_val"])):
        out = dirs["windows"] / split
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("window_*.npy"):
            f.unlink()
        n = 0
        for r in man[man.split == split].itertuples():
            x = pd.read_csv(dirs["inter"] / f"room_{r.room}" / r.split / f"{r.capture_id}.csv").to_numpy()
            W, starts = make_windows(x, L, stride)
            if len(W) == 0:
                skipped.append(dict(capture_id=r.capture_id, room=r.room, split=split, n_valid=len(x)))
                continue
            for w, s in zip(W, starts):
                wid = f"window_{n:06d}"
                np.save(out / f"{wid}.npy", w)
                index.append(dict(window_id=wid, split=split, capture_id=r.capture_id, room=r.room,
                                  label=r.label, start=s, length=L, stride=stride, preproc_hash=h,
                                  source_file=r.source_file))
                n += 1
        print(f"[windows] {split}: {n} windows")
    pd.DataFrame(index).to_csv(dirs["windows"] / "index.csv", index=False)
    pd.DataFrame(skipped, columns=["capture_id", "room", "split", "n_valid"]).to_csv(
        dirs["windows"] / "skipped_captures.csv", index=False)
    if skipped:
        print(f"[windows] WARNING: {len(skipped)} captures shorter than {L} valid observations were skipped "
              f"(see skipped_captures.csv)")


def load_split(cfg, split):
    """Load windows of a split. Returns (X float32 (n,150,256), meta DataFrame)."""
    dirs = workspace_dirs(cfg)
    idx = pd.read_csv(dirs["windows"] / "index.csv")
    meta = idx[idx.split == split].reset_index(drop=True)
    X = np.stack([np.load(dirs["windows"] / split / f"{w}.npy") for w in meta.window_id])
    return X, meta
