"""Config loading + preprocessing hash used for provenance."""
import hashlib, json, os, tempfile
from pathlib import Path
import yaml

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx.physical_cores_cache = os.cpu_count() or 4
except Exception:
    pass

PREPROC_KEYS = ("decode", "qc", "split", "window", "normalization")


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("workspace", "results"):
        cfg["paths"][k] = str(Path(cfg["paths"][k]).expanduser())
    return cfg


def preprocessing_hash(cfg):
    blob = json.dumps({k: cfg[k] for k in PREPROC_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def workspace_dirs(cfg):
    ws = Path(cfg["paths"]["workspace"])
    return dict(root=ws, inter=ws / "intermediate_csv", windows=ws / "windows",
                features=ws / "features", artifacts=ws / "artifacts")
