"""Cross-room few-label adaptation on frozen representations.

For target room T and source room S:
  * label set(T) = shared {SIT,STAND,WALK} + target-exclusive activities
  * shared classes: trained from S's labelled DEVELOPMENT windows only (no T labels)
  * exclusive classes: trained from K labelled T development CAPTURES per class (support set)
  * per-room embedding standardization (unsupervised domain alignment): each room's embeddings are
    z-scored with the mean/std of that room's unlabeled development windows.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .labels import EXCLUSIVE, ROOM_LABELS, SHARED, other_room


def room_stats(E_dev, rooms):
    st = {}
    for r in ("A", "B"):
        e = E_dev[np.asarray(rooms) == r]
        st[r] = (e.mean(0), np.maximum(e.std(0), 1e-6))
    return st


def standardize(E, stat):
    return (E - stat[0]) / stat[1]


class Head:
    def __init__(self, kind="logreg", C=1.0):
        self.kind, self.C = kind, C

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        if self.kind == "proto":
            self.protos = np.stack([X[y == c].mean(0) for c in self.classes_])
        else:
            self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
            self.clf.fit(X, y)
            self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        if self.kind == "proto":
            d = ((X[:, None] - self.protos[None]) ** 2).sum(-1)
            d = d - d.min(1, keepdims=True)
            p = np.exp(-d / (d.mean() + 1e-9)); return p / p.sum(1, keepdims=True)
        return self.clf.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(1)], p.max(1)


def sample_support(meta_dev, target, K, rng):
    """K distinct DEVELOPMENT captures of the target room for each target-exclusive activity."""
    sup = {}
    for lab in EXCLUSIVE[target]:
        caps = np.array(sorted(meta_dev[(meta_dev.room == target) & (meta_dev.label == lab)].capture_id.unique()))
        sup[lab] = list(rng.choice(caps, size=min(K, len(caps)), replace=False))
    return sup


def source_set(E_dev, meta_dev, stats, target):
    """Source-room labelled development windows of the shared activities (standardized)."""
    s = other_room(target)
    m = ((meta_dev.room == s) & meta_dev.label.isin(SHARED)).to_numpy()
    return standardize(E_dev[m], stats[s]), meta_dev.label.to_numpy()[m]


def fit_target_head(E_dev, meta_dev, stats, target, support, adapt_cfg):
    Xs, ys = source_set(E_dev, meta_dev, stats, target)
    caps = {c: lab for lab, cs in support.items() for c in cs}
    m = meta_dev.capture_id.isin(caps).to_numpy()
    assert (meta_dev.room.to_numpy()[m] == target).all() and (meta_dev.split.to_numpy()[m] == "development").all()
    Xt = standardize(E_dev[m], stats[target])
    yt = meta_dev.capture_id[m].map(caps).to_numpy()
    return Head(adapt_cfg["head"], adapt_cfg["C"]).fit(np.concatenate([Xs, Xt]), np.concatenate([ys, yt]))
