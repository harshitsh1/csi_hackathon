import numpy as np


def make_windows(x, length, stride):
    """Deterministic windows of `length` consecutive valid observations starting at 0 with the
    given stride. Never crosses captures (x is a single capture). Returns (W, starts)."""
    n = len(x)
    if n < length:
        return np.zeros((0, length, x.shape[1]), np.float32), []
    starts = list(range(0, n - length + 1, stride))
    return np.stack([x[s:s + length] for s in starts]).astype(np.float32), starts
