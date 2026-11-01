"""Damage-severity photo pipeline.

    explore : load labels -> validate (unreadable, duplicates, leakage) -> charts + summary
    train   : split -> ResNet18 features (cached) -> tune logistic regression with CV on train
              -> evaluate on test -> save run folder
    README  : the AUTO:image_eda and AUTO:image_results sections are refreshed

Run everything:         python -m src.image_pipeline.run
Only explore the data:  python -m src.image_pipeline.run --stage explore
"""
import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from src.image_pipeline import steps


def main(config_path: str = "configs/image_severity.yaml", stage: str = "all") -> Path:
    cfg = yaml.safe_load(open(config_path))
    run_dir = Path("runs") / f"{cfg['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")], force=True)
    log = logging.getLogger("image_pipeline")
    log.info("Run folder: %s", run_dir)

    # ---------------- stage 1: explore
    df = steps.load_labels(cfg)
    df, issues = steps.validate(df)
    eda_dir = Path("reports/image_eda")
    steps.explore(df, issues, cfg, eda_dir)
    shutil.copytree(eda_dir, run_dir / "eda", dirs_exist_ok=True)

    if stage == "all":
        # ---------------- stage 2: train + evaluate
        train, test, test_name = steps.split(df, cfg)
        cache = Path("data/feature_cache")
        X_tr = steps.extract_features(train["path"].tolist(), cfg, cache)
        X_te = steps.extract_features(test["path"].tolist(), cfg, cache)
        model, best, cv_score, cv = steps.tune_and_train(X_tr, train["label"], cfg)
        metrics, pred, proba = steps.evaluate(model, X_te, test["label"], train["label"], cfg, test_name)
        metrics.update(best_params=best, cv_score=cv_score, cv_scoring=cfg["tuning"]["scoring"])
        steps.save_artifacts(run_dir, model, metrics, cv, test, pred, proba,
                             model.named_steps["logreg"].classes_, cfg)

        # runs/ stays off GitHub: copy what the README shows into reports/
        Path("models").mkdir(exist_ok=True)
        shutil.copy(run_dir / "model.joblib", "models/image_severity_model.joblib")
        figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
        for old in figs.glob("image_*.png"):          # no stale charts from an earlier run
            old.unlink()
        for f in (run_dir / "figures").glob("*.png"):
            shutil.copy(f, figs / f"image_{f.name}")
        shutil.copy(run_dir / "metrics.json", "reports/image_metrics.json")
        log.info("Summary: %s", json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "recall_by_class"]}))

    from src.update_readme import main as update_readme
    update_readme()
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/image_severity.yaml")
    ap.add_argument("--stage", choices=["explore", "all"], default="all")
    a = ap.parse_args()
    main(a.config, a.stage)
