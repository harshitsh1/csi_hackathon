"""Run: pytest -q tests   (uses synthetic data only, plus real files in tests/ if you drop any there)"""
import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from src.csi_decode import unpack_bcm4366c0
from src.cleaning import clean_capture
from src.windowing import make_windows
from src.normalization import AmplitudeNormalizer
from src.data_index import index_room_b
from make_mock_data import encode_words

QC = dict(drop_nonfinite=True, drop_all_zero=True, max_zero_fraction=0.5, drop_consecutive_duplicates=True)


def test_decode_roundtrip_up_to_packet_scale():
    rng = np.random.default_rng(0)
    amp = rng.uniform(5, 900, (4, 256))
    dec = np.abs(unpack_bcm4366c0(encode_words(amp, rng)))
    ratio = dec / amp                                   # autoscale = one constant per packet
    assert np.all(np.std(ratio, 1) / np.mean(ratio, 1) < 0.02)


def test_decode_zero_packet_is_zero():
    assert np.all(unpack_bcm4366c0(np.zeros((1, 256), np.uint32)) == 0)


def test_cleaning_drops_bad_rows_and_keeps_order():
    x = np.random.default_rng(1).uniform(1, 9, (10, 256))
    x[2] = 0; x[4, 3] = np.nan; x[6] = x[5]
    c, r = clean_capture(x, QC)
    assert len(c) == 7 and r["n_all_zero"] == 1 and r["n_nonfinite"] == 1 and r["n_duplicate"] == 1
    assert np.array_equal(c[0], x[0]) and np.array_equal(c[-1], x[-1])


def test_windows_never_cross_capture_and_are_deterministic():
    x = np.arange(400 * 256, dtype=float).reshape(400, 256)
    W, s = make_windows(x, 150, 75)
    assert W.shape == (4, 150, 256) and s == [0, 75, 150, 225]
    assert np.array_equal(W[1][0], x[75]) and make_windows(x[:100], 150, 75)[0].shape[0] == 0


def test_normalizer_fit_is_dev_only_and_frozen():
    rng = np.random.default_rng(2)
    dev, val = rng.uniform(0, 500, (6, 150, 256)), rng.uniform(0, 5000, (3, 150, 256))
    n = AmplitudeNormalizer().fit(dev); m0 = n.mean.copy()
    n.transform(val); assert np.array_equal(m0, n.mean)


def test_room_b_split_is_seeded_capture_level():
    with tempfile.TemporaryDirectory() as d:
        for lab in ("Empty_Amp", "Lying_Amp"):
            for i in range(1, 41):
                (Path(d) / f"{lab}_{i}.csv").write_text("x\n")
        a, b = index_room_b(d, 25, 42), index_room_b(d, 25, 42)
        assert [r.split for r in a] == [r.split for r in b]
        assert sum(r.split == "validation" for r in a) == 50
        assert len({r.capture_id for r in a}) == 80
