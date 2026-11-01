"""Run the full crash text-classification pipeline:
    load -> validate & clean -> split -> build model -> tune (CV on train) -> evaluate on test
         -> explain -> save everything to runs/<run_name>_<timestamp>/

Run: python -m src.pipeline.run --config configs/text_crash.yaml
"""
import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from src.pipeline import steps


def main(config_path: str) -> Path:
    cfg = yaml.safe_load(open(config_path))
    run_dir = Path("runs") / f"{cfg['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")          # reproducibility

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")],
                        force=True)
    log = logging.getLogger("pipeline")
    log.info("Run folder: %s", run_dir)

    df = steps.load_data(cfg)
    df = steps.validate_and_clean(df, cfg)
    X_tr, X_te, y_tr, y_te = steps.split(df, cfg)
    model = steps.build_model(cfg)
    model, best_params, cv_score, cv_table = steps.tune_and_train(model, X_tr, y_tr, cfg)
    metrics, proba = steps.evaluate(model, X_tr, y_tr, X_te, y_te, cfg)
    metrics.update(best_params={k: list(v) if isinstance(v, tuple) else v for k, v in best_params.items()},
                   cv_score=cv_score, cv_scoring=cfg["tuning"]["scoring"])
    words = steps.top_words(model, cfg["evaluation"]["top_n_words"])
    steps.save_artifacts(run_dir, model, metrics, words, cv_table, X_te, y_te, proba, cfg)

    # Keep a pointer to the latest run for predict.py and the README
    Path("models").mkdir(exist_ok=True)
    shutil.copy(run_dir / "model.joblib", "models/text_crash_model.joblib")
    # runs/ stays off GitHub; copy the latest charts and metrics to reports/ for the README
    figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
    for f in (run_dir / "figures").glob("*.png"):
        shutil.copy(f, figs / f"text_{f.name}")
    shutil.copy(run_dir / "metrics.json", "reports/text_metrics.json")
    shutil.copy(run_dir / "top_words.json", "reports/text_top_words.json")
    log.info("Summary: %s", json.dumps({k: metrics[k] for k in
                                        ["roc_auc", "precision", "recall", "baseline_accuracy_majority_class"]}))
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/text_crash.yaml")
    main(ap.parse_args().config)
