"""Tasks 1-3 evaluated on the fixed VALIDATION windows. All fitting uses DEVELOPMENT data only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

from . import adapt
from .features import fingerprint_features
from .labels import EXCLUSIVE, ROOM_LABELS, SHARED, UNIFIED_LABELS

REPS = ["ssl", "random", "hand"]


class Results:
    def __init__(self):
        self.rows = []

    def add(self, experiment, target_room, K, metric, value, method=""):
        self.rows.append(dict(experiment=experiment, target_room=target_room, K=K, metric=metric,
                              value=float(value), method=method))

    def df(self):
        return pd.DataFrame(self.rows)


def _mf1(y, p, labels):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


# ---------------------------------------------------------------- Task 1: room fingerprint
def task1(Xn_dev, meta_dev, Xn_val, meta_val, cfg, R):
    lens = cfg["fingerprint"]["window_lengths"]
    Fd = np.concatenate([fingerprint_features(Xn_dev, L) for L in lens])
    yd = np.tile(meta_dev.room.to_numpy(), len(lens))
    sc = StandardScaler().fit(Fd)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Fd), yd)
    yv = meta_val.room.to_numpy()
    for L in lens:
        pv = clf.predict(sc.transform(fingerprint_features(Xn_val, L)))
        R.add("task1_room_id", "all", 0, f"room_accuracy_L{L}", (pv == yv).mean(), "fingerprint_lr")
    pv = clf.predict(sc.transform(fingerprint_features(Xn_val, 150)))
    R.add("task1_room_id", "all", 0, "room_accuracy", (pv == yv).mean(), "fingerprint_lr")
    for r in "AB":
        R.add("task1_room_id", r, 0, "room_accuracy", (pv[yv == r] == r).mean(), "fingerprint_lr")
    return dict(scaler=sc, clf=clf)


def route(router, Xn, L=150):
    p = router["clf"].predict_proba(router["scaler"].transform(fingerprint_features(Xn, L)))
    return router["clf"].classes_[p.argmax(1)], p.max(1)


# ---------------------------------------------------------------- Task 2: frozen-encoder linear probe
def task2(feats, meta_dev, meta_val, R):
    yd, yv = meta_dev.label.to_numpy(), meta_val.label.to_numpy()
    for rep in REPS:
        sc = StandardScaler().fit(feats[rep]["dev"])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
            sc.transform(feats[rep]["dev"]), yd)
        pv = clf.predict(sc.transform(feats[rep]["val"]))
        R.add("task2_linear_probe", "all", 0, "macro_f1", _mf1(yv, pv, UNIFIED_LABELS), rep)
        R.add("task2_linear_probe", "all", 0, "balanced_accuracy", balanced_accuracy_score(yv, pv), rep)
        for r in "AB":
            m = meta_val.room.to_numpy() == r
            R.add("task2_linear_probe", r, 0, "macro_f1", _mf1(yv[m], pv[m], ROOM_LABELS[r]), rep)


# ---------------------------------------------------------------- Task 3: few-label cross-room adaptation
def _metrics(R, rep, target, K, y, p, per_draw):
    labs = ROOM_LABELS[target]
    per_draw.setdefault((rep, target, K), []).append(dict(
        macro_f1=_mf1(y, p, labs), balanced_accuracy=balanced_accuracy_score(y, p),
        target_exclusive_macro_f1=_mf1(y, p, EXCLUSIVE[target]),
        shared_macro_f1=_mf1(y, p, SHARED)))


def task3(feats, meta_dev, meta_val, router, Xn_val, cfg, R, keep_k=5):
    """Returns dict with predictions (for confusion matrices) and the fitted default state."""
    a, per_draw, kept = cfg["adapt"], {}, {}
    route_rows = {}
    for rep in REPS:
        Ed, Ev = feats[rep]["dev"], feats[rep]["val"]
        stats = adapt.room_stats(Ed, meta_dev.room.to_numpy())
        for K in a["K"]:
            for d in range(a["n_draws"]):
                rng = np.random.default_rng(cfg["seed"] * 1000 + K * 10 + d)
                heads = {}
                for T in "AB":
                    sup = adapt.sample_support(meta_dev, T, K, rng)
                    heads[T] = adapt.fit_target_head(Ed, meta_dev, stats, T, sup, a)
                    mv = (meta_val.room == T).to_numpy()
                    p, conf = heads[T].predict(adapt.standardize(Ev[mv], stats[T]))
                    _metrics(R, rep, T, K, meta_val.label.to_numpy()[mv], p, per_draw)
                    if rep == "ssl" and K == keep_k and d == 0:
                        kept[T] = dict(y=meta_val.label.to_numpy()[mv], p=p)
                if rep == "ssl":
                    _routing(Ev, meta_val, Xn_val, heads, stats, router, K, route_rows)
    for (rep, T, K), lst in per_draw.items():
        for m in lst[0]:
            v = np.array([x[m] for x in lst])
            R.add("task3_adaptation", T, K, m, v.mean(), rep)
            R.add("task3_adaptation", T, K, m + "_std", v.std(), rep)
    for (K, mode), lst in route_rows.items():
        for m in lst[0]:
            v = np.array([x[m] for x in lst])
            R.add("task3_routing", "pooled", K, m, v.mean(), mode)
    return kept


def _routing(Ev, meta_val, Xn_val, heads, stats, router, K, rows):
    """Pooled A+B validation windows, unified 8-label space. Modes:
       predicted   : router picks which room's adapted head (and room standardization) to use
       oracle      : true room picks the head
       none        : both heads run on every window (each with its own room standardization);
                     the more confident prediction wins (same heads, no routing stage)."""
    y, n = meta_val.label.to_numpy(), len(meta_val)
    room_pred, _ = route(router, Xn_val)
    true_room = meta_val.room.to_numpy()
    P = {T: heads[T].predict(adapt.standardize(Ev, stats[T])) for T in "AB"}   # (labels, conf) per head
    def pick(rooms):
        lab = np.where(rooms == "A", P["A"][0], P["B"][0]); return lab
    none_lab = np.where(P["A"][1] >= P["B"][1], P["A"][0], P["B"][0])
    for mode, pred in (("predicted_room_routing", pick(room_pred)), ("oracle_room", pick(true_room)),
                       ("no_routing", none_lab)):
        rows.setdefault((K, mode), []).append(dict(
            macro_f1=_mf1(y, pred, UNIFIED_LABELS), balanced_accuracy=balanced_accuracy_score(y, pred),
            accuracy=(pred == y).mean()))
    rows.setdefault((K, "predicted_room_routing"), [])[-1]["routing_room_accuracy"] = (room_pred == true_room).mean()
    rows.setdefault((K, "oracle_room"), [])[-1]["routing_room_accuracy"] = 1.0
    rows.setdefault((K, "no_routing"), [])[-1]["routing_room_accuracy"] = float("nan")


def confusion_tables(kept, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for T, d in kept.items():
        labs = ROOM_LABELS[T]
        cm = confusion_matrix(d["y"], d["p"], labels=labs)
        pd.DataFrame(cm, index=labs, columns=labs).to_csv(out_dir / f"confusion_K5_target{T}.csv")
        p, r, f, s = precision_recall_fscore_support(d["y"], d["p"], labels=labs, zero_division=0)
        pd.DataFrame(dict(activity=labs, precision=p, recall=r, f1=f, support=s)).to_csv(
            out_dir / f"per_class_K5_target{T}.csv", index=False)
