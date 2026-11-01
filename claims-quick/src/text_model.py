"""NLP: flag complaints that describe a crash, from the text alone.

Model: TF-IDF (turns each description into word weights) + logistic regression.
Run: python -m src.text_model
"""
import json
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
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

REPORTS, FIGS, MODELS = Path("reports"), Path("reports/figures"), Path("models")


def main() -> None:
    df = pd.read_csv("data/complaints.csv")
    X, y = df["description"].str.lower(), df["crash"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(classification_report(y_te, pred, target_names=["no crash", "crash"], digits=3))

    # Baseline to beat: always predict the most common class
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    metrics = {
        "n_train": len(X_tr), "n_test": len(X_te), "crash_rate": round(float(y.mean()), 3),
        "auc": round(roc_auc_score(y_te, proba), 3),
        "recall_crash": round(recall_score(y_te, pred), 3),
        "precision_crash": round(precision_score(y_te, pred), 3),
        "baseline_accuracy_always_no_crash": round(float((dummy.predict(X_te) == y_te).mean()), 3),
        "accuracy": round(float((pred == y_te).mean()), 3),
    }
    print(json.dumps(metrics, indent=2))

    # The words that push the prediction most toward "crash" and "no crash"
    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    terms, coef = np.array(vec.get_feature_names_out()), clf.coef_[0]
    top_crash = terms[np.argsort(coef)[-15:]][::-1].tolist()
    top_safe = terms[np.argsort(coef)[:15]].tolist()
    print("Words pointing to CRASH:   ", ", ".join(top_crash))
    print("Words pointing to NO CRASH:", ", ".join(top_safe))
    metrics["top_words_crash"], metrics["top_words_no_crash"] = top_crash, top_safe

    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    path = REPORTS / "metrics.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm["text_crash_classifier"] = metrics
    path.write_text(json.dumps(allm, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["no crash", "crash"])
    plt.title("Crash flag from text: test set")
    plt.savefig(FIGS / "text_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save 10 mistakes to read for the README
    test = pd.DataFrame({"description": X_te, "true": y_te, "p_crash": proba.round(3)})
    wrong = test[(test["p_crash"] >= 0.5) != (test["true"] == 1)]
    wrong.sort_values("p_crash").to_csv(REPORTS / "text_errors.csv", index=False)
    joblib.dump(model, MODELS / "text_crash_model.joblib")


if __name__ == "__main__":
    main()
