"""log1p + per-subcarrier z-score. Statistics are fitted on DEVELOPMENT windows only
(pooled across both rooms, no labels used) and then frozen."""
import numpy as np


class AmplitudeNormalizer:
    def __init__(self, log1p=True, std_floor=0.05):
        self.log1p, self.std_floor = log1p, std_floor
        self.mean = self.std = None

    def _f(self, X):
        return np.log1p(X) if self.log1p else X

    def fit(self, X, chunk=256):
        n = s = s2 = 0
        for i in range(0, len(X), chunk):
            a = self._f(X[i:i + chunk].astype(np.float64)).reshape(-1, X.shape[-1])
            n += len(a); s = s + a.sum(0); s2 = s2 + (a ** 2).sum(0)
        self.mean = s / n
        self.std = np.maximum(np.sqrt(np.maximum(s2 / n - self.mean ** 2, 0)), self.std_floor)
        return self

    def transform(self, X):
        out = np.empty(X.shape, np.float32)
        for i in range(0, len(X), 256):
            out[i:i + 256] = ((self._f(X[i:i + 256].astype(np.float64)) - self.mean) / self.std)
        return out

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, log1p=self.log1p, std_floor=self.std_floor)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        o = cls(bool(d["log1p"]), float(d["std_floor"])); o.mean, o.std = d["mean"], d["std"]
        return o
