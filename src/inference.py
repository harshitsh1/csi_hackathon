"""Shared inference path used by run_episode.py and the validation-prediction export."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from . import adapt
from .config import workspace_dirs
from .encoder import CSIEncoder, embed
from .ingest import load_capture
from .cleaning import clean_capture
from .labels import EXCLUSIVE
from .normalization import AmplitudeNormalizer
from .tasks import route
from .windowing import make_windows


def predict_from_embeddings(state, Xn, E):
    room, _ = route(state["router"], Xn)
    labels, conf = np.empty(len(E), dtype=object), np.zeros(len(E))
    for T in "AB":
        m = room == T
        if m.any():
            p, c = state["heads"][T].predict(adapt.standardize(E[m], state["stats"][T]))
            labels[m], conf[m] = p, c
    return room, labels, conf


def load_artifacts(cfg, device="cpu"):
    d = workspace_dirs(cfg)["artifacts"]
    art = joblib.load(d / "heads.joblib")
    art["norm"] = AmplitudeNormalizer.load(d / "normalizer.npz")
    c = cfg["ssl"]
    m = CSIEncoder(256, c["feat_dim"], c["proj_dim"]); m.load_state_dict(torch.load(d / "encoder.pt", map_location=device))
    art["model"], art["device"] = m.eval(), device
    art["bank_emb"] = np.load(d / "bank_emb.npy"); art["bank_meta"] = pd.read_csv(d / "bank_meta.csv")
    return art


def capture_windows(path, cfg, stride=None, pad_short=True):
    """Raw capture (.pcap -> Room-A decoder, .csv -> amplitude CSV) -> (n,150,256) windows. The extension
    only selects the file parser; it is never given to a model."""
    path = Path(path)
    rec = dict(room="A" if path.suffix.lower() == ".pcap" else "B", path=str(path))
    amp, _ = load_capture(rec, cfg)
    clean, _ = clean_capture(amp, cfg["qc"])
    L = cfg["window"]["length"]
    if len(clean) == 0:
        return np.zeros((0, L, 256), np.float32), "empty"
    note = ""
    if len(clean) < L:
        if not pad_short:
            return np.zeros((0, L, 256), np.float32), f"only {len(clean)} valid observations"
        clean = np.concatenate([clean, np.repeat(clean[-1:], L - len(clean), 0)])
        note = "padded"
    W, _ = make_windows(clean, L, stride or L)
    return W, note


def episode_windows(episode_dir, cfg, pad_short=True):
    ep = Path(episode_dir); X, ids, notes = [], [], []
    for f in sorted(ep.rglob("*")):
        if f.name.startswith(".") or "__MACOSX" in f.parts:
            continue
        if f.suffix == ".npy":
            w = np.load(f).astype(np.float32)
            w = w[None] if w.ndim == 2 else w
            X.append(w); ids += [f.stem] if len(w) == 1 else [f"{f.stem}_w{i}" for i in range(len(w))]
        elif f.suffix.lower() in (".pcap", ".csv") and f.name != "support.csv":
            W, note = capture_windows(f, cfg, pad_short=pad_short)
            if note:
                notes.append((f.name, note))
            if len(W):
                X.append(W); ids += [f"{f.stem}_w{i}" for i in range(len(W))]
    return (np.concatenate(X) if X else np.zeros((0, 150, 256), np.float32)), ids, notes


def rebuild_heads(art, cfg, support_df, base_dir):
    """Refit both target heads using the episode's own labelled support captures."""
    norm, model, dev = art["norm"], art["model"], art["device"]
    stats, a = art["stats"], cfg["adapt"]
    meta, E = art["bank_meta"], art["bank_emb"]
    for T in "AB":
        s = support_df[(support_df.room == T) & support_df.label.str.upper().isin(EXCLUSIVE[T])]
        if s.empty:
            continue
        Xt, yt = [], []
        for r in s.itertuples():
            p = Path(r.path); p = p if p.is_absolute() else Path(base_dir) / p
            W, _ = capture_windows(p, cfg, stride=cfg["window"]["stride_dev"])
            Xt.append(W); yt += [r.label.upper()] * len(W)
        Et = adapt.standardize(embed(model, norm.transform(np.concatenate(Xt)), dev), stats[T])
        Xs, ys = adapt.source_set(E, meta, stats, T)
        art["heads"][T] = adapt.Head(a["head"], a["C"]).fit(np.concatenate([Xs, Et]), np.concatenate([ys, yt]))
    return art
