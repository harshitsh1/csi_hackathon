#!/usr/bin/env python
"""Process one challenge episode end-to-end and write predictions (window_id, room, label, confidence).

  python run_episode.py --episode_dir EPISODE/ --out predictions.csv [--support_csv support.csv]

EPISODE/ may hold raw captures (.pcap = Room-A format, .csv = Room-B amplitude format) and/or
window_*.npy files of shape (150,256). The optional support CSV (columns: path,room,label) gives K labelled
target captures for target-exclusive activities; heads are then re-fitted from them. Without it, the default
heads trained by run_pipeline.py (K = adapt.default_K, drawn from development captures) are used."""
import argparse
import pandas as pd
from src.config import load_config
from src.inference import episode_windows, load_artifacts, predict_from_embeddings, rebuild_heads
from src.encoder import embed
from src.pipeline import pick_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--episode_dir", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--support_csv")
    p.add_argument("--device", default="auto")
    p.add_argument("--no_pad", action="store_true", help="skip captures with <150 valid observations instead of padding")
    a = p.parse_args()
    cfg = load_config(a.config)
    art = load_artifacts(cfg, pick_device(a.device))
    if a.support_csv:
        sup = pd.read_csv(a.support_csv)
        art = rebuild_heads(art, cfg, sup, __import__("pathlib").Path(a.support_csv).parent)
    X, ids, notes = episode_windows(a.episode_dir, cfg, pad_short=not a.no_pad)
    for n in notes:
        print("[episode] note:", *n)
    Xn = art["norm"].transform(X)
    E = embed(art["model"], Xn, art["device"])
    room, label, conf = predict_from_embeddings(art, Xn, E)
    pd.DataFrame(dict(window_id=ids, room=room, label=label, confidence=conf.round(4))).to_csv(a.out, index=False)
    print(f"[episode] wrote {len(ids)} predictions to {a.out}")


if __name__ == "__main__":
    main()
