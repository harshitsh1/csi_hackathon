#!/usr/bin/env python
"""Full pipeline: ingest -> windows -> train (SSL) -> tasks.   python run_pipeline.py --config configs/experiment.yaml --stage all"""
import os, sys, warnings
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx.physical_cores_cache = os.cpu_count() or 4
except Exception:
    pass

import argparse
from src.config import load_config
from src import ingest, pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--stage", default="all", choices=["ingest", "windows", "train", "tasks", "all"])
    p.add_argument("--data_root", help="override paths.data_root")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.data_root:
        cfg["paths"]["data_root"] = a.data_root
    dev = pipeline.pick_device(a.device)
    if a.stage in ("ingest", "all"):
        ingest.run_ingest(cfg, a.workers)
    if a.stage in ("windows", "all"):
        ingest.run_windows(cfg)
    if a.stage in ("train", "all"):
        pipeline.run_train(cfg, dev)
    if a.stage in ("tasks", "all"):
        pipeline.run_tasks(cfg, dev)


if __name__ == "__main__":
    main()
