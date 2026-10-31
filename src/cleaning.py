import numpy as np


def clean_capture(amp, qc):
    """Row-wise quality control, preserving temporal order. Returns (clean, report)."""
    rep = dict(n_in=int(len(amp)), n_nonfinite=0, n_all_zero=0, n_mostly_zero=0, n_duplicate=0)
    keep = np.ones(len(amp), bool)
    if qc["drop_nonfinite"]:
        bad = ~np.isfinite(amp).all(axis=1)
        rep["n_nonfinite"] = int(bad.sum()); keep &= ~bad
    a = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
    if qc["drop_all_zero"]:
        z = (a == 0).all(axis=1) & keep
        rep["n_all_zero"] = int(z.sum()); keep &= ~z
    zf = qc.get("max_zero_fraction")
    if zf is not None:
        mz = ((a == 0).mean(axis=1) > zf) & keep
        rep["n_mostly_zero"] = int(mz.sum()); keep &= ~mz
    out = amp[keep]
    if qc["drop_consecutive_duplicates"] and len(out) > 1:
        dup = np.r_[False, (out[1:] == out[:-1]).all(axis=1)]
        rep["n_duplicate"] = int(dup.sum()); out = out[~dup]
    rep["n_out"] = int(len(out))
    return out, rep
