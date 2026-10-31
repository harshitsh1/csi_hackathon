"""Stage 3 (representation learning) and stage 4 (tasks + figures + artifacts)."""
import joblib
import numpy as np
import pandas as pd
import torch

from . import adapt, figures, tasks
from .config import workspace_dirs
from .encoder import CSIEncoder, embed, random_encoder, train_ssl
from .features import handcrafted_features
from .ingest import load_split
from .normalization import AmplitudeNormalizer


def pick_device(arg="auto"):
    return ("cuda" if torch.cuda.is_available() else "cpu") if arg == "auto" else arg


def _norm(cfg, X_dev):
    n = cfg["normalization"]
    return AmplitudeNormalizer(n["log1p"], n["std_floor"]).fit(X_dev)       # DEVELOPMENT data only


def run_train(cfg, device="cpu"):
    dirs = workspace_dirs(cfg)
    for k in ("features", "artifacts"):
        dirs[k].mkdir(parents=True, exist_ok=True)
    X_dev, meta_dev = load_split(cfg, "development")
    X_val, meta_val = load_split(cfg, "validation")
    print(f"[train] dev windows {X_dev.shape}, val windows {X_val.shape}")
    norm = _norm(cfg, X_dev)
    norm.save(dirs["artifacts"] / "normalizer.npz")
    Xn_dev, Xn_val = norm.transform(X_dev), norm.transform(X_val)
    del X_dev, X_val
    # --- self-supervised pretraining: dev windows from both rooms, NO activity or room labels passed in
    model, hist = train_ssl(Xn_dev, cfg, device, cfg["seed"])
    torch.save(model.state_dict(), dirs["artifacts"] / "encoder.pt")
    pd.DataFrame(dict(epoch=range(1, len(hist) + 1), loss=hist)).to_csv(dirs["artifacts"] / "ssl_loss.csv", index=False)
    rnd = random_encoder(cfg, Xn_dev.shape[-1], seed=0)
    for split, Xn in (("dev", Xn_dev), ("val", Xn_val)):
        np.save(dirs["features"] / f"ssl_{split}.npy", embed(model, Xn, device))
        np.save(dirs["features"] / f"random_{split}.npy", embed(rnd, Xn, device))
        np.save(dirs["features"] / f"hand_{split}.npy", handcrafted_features(Xn))
    print("[train] features cached")


def run_tasks(cfg, device="cpu"):
    dirs = workspace_dirs(cfg)
    res_dir = __import__("pathlib").Path(cfg["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)
    X_dev, meta_dev = load_split(cfg, "development")
    X_val, meta_val = load_split(cfg, "validation")
    norm = AmplitudeNormalizer.load(dirs["artifacts"] / "normalizer.npz")
    Xn_dev, Xn_val = norm.transform(X_dev), norm.transform(X_val)
    del X_dev, X_val
    feats = {r: dict(dev=np.load(dirs["features"] / f"{r}_dev.npy"), val=np.load(dirs["features"] / f"{r}_val.npy"))
             for r in tasks.REPS}
    assert len(feats["ssl"]["dev"]) == len(meta_dev) and len(feats["ssl"]["val"]) == len(meta_val), \
        "feature cache out of date: rerun --stage train"
    R = tasks.Results()
    router = tasks.task1(Xn_dev, meta_dev, Xn_val, meta_val, cfg, R)
    print("[tasks] task 1 done"); tasks.task2(feats, meta_dev, meta_val, R)
    print("[tasks] task 2 done"); kept = tasks.task3(feats, meta_dev, meta_val, router, Xn_val, cfg, R)
    print("[tasks] task 3 done")
    df = R.df(); df.to_csv(res_dir / "results.csv", index=False)
    tasks.confusion_tables(kept, res_dir)
    figures.label_efficiency(df, res_dir / "fig_label_efficiency.png")
    figures.fingerprint(df, res_dir / "fig_environment_fingerprint.png")
    figures.routing(df, res_dir / "fig_routing_contribution.png")
    figures.confusion(res_dir, res_dir / "fig_confusion_K5.png")
    figures.embedding_plot(feats["ssl"]["val"], meta_val, res_dir / "fig_encoder_embedding.png")
    # ---- artifacts for run_episode.py (default heads use support drawn from DEVELOPMENT captures only)
    a = cfg["adapt"]; stats = adapt.room_stats(feats["ssl"]["dev"], meta_dev.room.to_numpy())
    rng = np.random.default_rng(cfg["seed"]); heads = {}
    for T in "AB":
        heads[T] = adapt.fit_target_head(feats["ssl"]["dev"], meta_dev, stats, T,
                                         adapt.sample_support(meta_dev, T, a["default_K"], rng), a)
    joblib.dump(dict(router=router, stats=stats, heads=heads), dirs["artifacts"] / "heads.joblib")
    np.save(dirs["artifacts"] / "bank_emb.npy", feats["ssl"]["dev"])
    meta_dev[["window_id", "capture_id", "room", "label", "split"]].to_csv(dirs["artifacts"] / "bank_meta.csv", index=False)
    # validation predictions in the required output format (routed system, default K)
    from .inference import predict_from_embeddings
    room, lab, conf = predict_from_embeddings(dict(router=router, stats=stats, heads=heads),
                                              Xn_val, feats["ssl"]["val"])
    pd.DataFrame(dict(window_id=meta_val.window_id, room=room, label=lab, confidence=conf)).to_csv(
        res_dir / "val_predictions.csv", index=False)
    print(f"[tasks] results written to {res_dir}")
    return df
