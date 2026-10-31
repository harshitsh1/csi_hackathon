"""Hand-crafted baseline features and room-fingerprint features (operate on normalized windows)."""
import numpy as np


def handcrafted_features(Xn, n_bands=16):
    """(n,T,256) -> (n,128): per 16-subcarrier band mean/std/mean|diff|/range + 4 log spectral
    band energies (Doppler-like) of the band-averaged, detrended amplitude."""
    n, T, S = Xn.shape
    b = Xn.reshape(n, T, n_bands, S // n_bands).mean(3)
    mean, std = b.mean(1), b.std(1)
    mad = np.abs(np.diff(b, axis=1)).mean(1)
    rng = b.max(1) - b.min(1)
    P = np.abs(np.fft.rfft(b - mean[:, None], axis=1)) ** 2 / T
    nb = P.shape[1]
    edges = [1, max(2, int(.05 * nb)), max(3, int(.13 * nb)), max(4, int(.33 * nb)), nb]
    spec = [np.log1p(P[:, a:b_].sum(1)) for a, b_ in zip(edges[:-1], edges[1:])]
    return np.concatenate([mean, std, mad, rng, *spec], 1).astype(np.float32)


def fingerprint_features(Xn, L=None):
    """Static-channel fingerprint: per-subcarrier temporal mean and std over the first L observations."""
    x = Xn if L is None else Xn[:, :L]
    return np.concatenate([x.mean(1), x.std(1)], 1).astype(np.float32)
