"""The steps of the crash text-classification pipeline.

Each step is a small function with clear inputs and outputs, so it can be
tested and reused on its own. src/pipeline/run.py chains them together.
"""
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

log = logging.getLogger("pipeline")


# -------------------------------------------- 1. load
def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["path"])
    log.info("Loaded %s rows from %s", f"{len(df):,}", cfg["data"]["path"])
    return df


# ----------------------------------------- 2. validate + clean
def validate_and_clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    text, target = cfg["data"]["text_column"], cfg["data"]["target_column"]
    missing = {text, target} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(df[target].dropna().unique()) <= {0, 1}:
        raise ValueError(f"Target '{target}' must be 0/1")

    n0 = len(df)
    # drop missing values in target and text description
    df = df.dropna(subset=[text, target]).copy()
    df[text] = df[text].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    # drop very short descriptions that wouldn't add any value
    df = df[df[text].str.split().str.len() >= cfg["data"]["min_words"]]
    # drop duplicated text descriptions that would leak between test and train
    if cfg["data"]["drop_duplicate_text"]:
        df = df.drop_duplicates(subset=[text])
    # make sure the target is integer
    df[target] = df[target].astype(int)
    log.info("Cleaning kept %s of %s rows; positive rate %.1f%%",
             f"{len(df):,}", f"{n0:,}", 100 * df[target].mean())
    return df.reset_index(drop=True)


# ------------------------------------------ 3. split
def split(df: pd.DataFrame, cfg: dict):
    text, target = cfg["data"]["text_column"], cfg["data"]["target_column"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[text], df[target], test_size=cfg["split"]["test_size"],
        stratify=df[target], random_state=cfg["split"]["random_state"])
    log.info("Train %s / test %s (stratified)", f"{len(X_tr):,}", f"{len(X_te):,}")
    return X_tr, X_te, y_tr, y_te


# ------------------------------------------ 4. model definition
def build_model(cfg: dict) -> Pipeline:
    """One sklearn Pipeline: text -> TF-IDF features -> logistic regression.
    Because vectorizer and classifier are fitted together, cross-validation
    never lets test-fold words leak into the vocabulary."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**cfg["model"]["tfidf"])),
        ("logreg", LogisticRegression(**cfg["model"]["logreg"])),
    ])


# -------------------------------------------- 5. tune + train
def tune_and_train(model: Pipeline, X_tr, y_tr, cfg: dict):
    t = cfg["tuning"]
    grid = {k: [tuple(v) if isinstance(v, list) else v for v in vals] for k, vals in t["grid"].items()}
    search = GridSearchCV(
        model, grid, scoring=t["scoring"],
        cv=StratifiedKFold(t["cv_folds"], shuffle=True, random_state=cfg["split"]["random_state"]),
        n_jobs=-1, refit=True)
    search.fit(X_tr, y_tr)
    log.info("Best CV %s = %.3f with %s", t["scoring"], search.best_score_, search.best_params_)
    cv = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return search.best_estimator_, search.best_params_, round(float(search.best_score_), 4), cv


# -------------------------------------------- 6. evaluate
def evaluate(model: Pipeline, X_tr, y_tr, X_te, y_te, cfg: dict) -> tuple[dict, np.ndarray]:
    thr = cfg["evaluation"]["threshold"]
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "positive_rate_test": round(float(y_te.mean()), 4),
        "threshold": thr,
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "average_precision": round(average_precision_score(y_te, proba), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "accuracy": round(float((pred == y_te).mean()), 4),
        "baseline_accuracy_majority_class": round(float((dummy.predict(X_te) == y_te).mean()), 4),
    }
    # How recall and workload move with the threshold
    metrics["threshold_table"] = [
        {"threshold": t, "pct_flagged": round(100 * float((proba >= t).mean()), 1),
         "precision": round(precision_score(y_te, proba >= t, zero_division=0), 3),
         "recall": round(recall_score(y_te, proba >= t), 3)}
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    log.info("Test ROC-AUC %.3f | precision %.3f | recall %.3f",
             metrics["roc_auc"], metrics["precision"], metrics["recall"])
    return metrics, proba


# --------------------------------------------------- 7. explain
def top_words(model: Pipeline, n: int) -> dict:
    terms = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["logreg"].coef_[0]
    order = np.argsort(coef)
    return {"towards_crash": [(t, round(float(c), 3)) for t, c in zip(terms[order[-n:][::-1]], coef[order[-n:][::-1]])],
            "towards_no_crash": [(t, round(float(c), 3)) for t, c in zip(terms[order[:n]], coef[order[:n]])]}


# --------------------------------------------------- 8. save
def save_artifacts(run_dir: Path, model, metrics: dict, words: dict, cv: pd.DataFrame,
                   X_te, y_te, proba: np.ndarray, cfg: dict) -> None:
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "top_words.json").write_text(json.dumps(words, indent=2))
    cv.sort_values("rank_test_score").to_csv(run_dir / "cv_results.csv", index=False)

    thr = cfg["evaluation"]["threshold"]
    test = pd.DataFrame({"text": X_te.values, "true": y_te.values, "p_crash": proba.round(4)})
    test["error"] = np.where((test["p_crash"] >= thr) & (test["true"] == 0), "false positive",
                    np.where((test["p_crash"] < thr) & (test["true"] == 1), "false negative", ""))
    n = cfg["evaluation"]["n_errors_to_save"]
    errors = pd.concat([
        test[test["error"] == "false negative"].nsmallest(n // 2, "p_crash"),
        test[test["error"] == "false positive"].nlargest(n // 2, "p_crash")])
    errors.to_csv(run_dir / "errors_to_review.csv", index=False)

    figs = run_dir / "figures"
    ConfusionMatrixDisplay.from_predictions(y_te, proba >= thr, display_labels=["no crash", "crash"])
    plt.title(f"Confusion matrix (threshold {thr})"); plt.savefig(figs / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
    RocCurveDisplay.from_predictions(y_te, proba)
    plt.title("ROC curve, test set"); plt.savefig(figs / "roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()
    PrecisionRecallDisplay.from_predictions(y_te, proba)
    plt.title("Precision-recall curve, test set"); plt.savefig(figs / "pr_curve.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    top = words["towards_crash"][:15][::-1] + words["towards_no_crash"][:15][::-1]
    ax.barh([w for w, _ in top], [c for _, c in top],
            color=["#E45756" if c > 0 else "#4C78A8" for _, c in top])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title("Strongest words (logistic regression coefficients)")
    ax.set_xlabel("coefficient  (+ pushes towards crash)")
    fig.savefig(figs / "top_words.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved model, metrics, figures and errors to %s", run_dir)
