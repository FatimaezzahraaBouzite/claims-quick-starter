"""Steps of the damage-severity photo pipeline.

Each step is a small function with clear inputs and outputs.
src/image_pipeline/run.py chains them:  explore -> train & evaluate -> README.
"""
import hashlib
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("image_pipeline")
COLORS = {"minor": "#54A24B", "moderate": "#F58518", "severe": "#E45756"}


# ============================================== 1. load labels
def _parse_label_line(line: str) -> tuple[int, float]:
    """YOLO line -> (class id, box area as a share of the photo).
    Works for boxes ('c x y w h') and polygons ('c x1 y1 x2 y2 ...')."""
    parts = line.split()
    cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
    if len(coords) == 4:
        return cls, coords[2] * coords[3]
    xs, ys = coords[0::2], coords[1::2]
    return cls, (max(xs) - min(xs)) * (max(ys) - min(ys))


def load_labels(cfg: dict) -> pd.DataFrame:
    """One row per photo: split, path, label, number of damages, damaged area."""
    root = Path(cfg["data"]["root"])
    names = yaml.safe_load(open(root / "data.yaml"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    order = cfg["data"]["class_order"]

    def to_class(name: str) -> str:
        return next((o for o in order if o in name.lower()), name.lower())

    rows = []
    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lbl = root / split / "labels" / f"{img.stem}.txt"
            boxes = []
            if lbl.exists():
                boxes = [_parse_label_line(l) for l in lbl.read_text().splitlines() if l.strip()]
            classes = [to_class(names[c]) for c, _ in boxes]
            label = max(classes, key=lambda c: order.index(c) if c in order else -1) if classes else None
            rows.append({"split": split, "path": str(img), "label": label, "n_damages": len(boxes),
                         "damaged_area": round(min(1.0, sum(a for _, a in boxes)), 4),
                         "n_classes_in_photo": len(set(classes))})
    df = pd.DataFrame(rows)
    log.info("Found %s photos across splits %s", f"{len(df):,}", sorted(df["split"].unique()))
    return df


# =========================================== 2. validate
def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop unlabelled or unreadable photos; find duplicates across splits (leakage)."""
    issues = {"no_label": int(df["label"].isna().sum())}
    df = df.dropna(subset=["label"]).copy()

    sizes, hashes, bad = [], [], []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
            hashes.append(hashlib.md5(Path(p).read_bytes()).hexdigest())
        except Exception:
            sizes.append((None, None)); hashes.append(None); bad.append(p)
    df["width"], df["height"] = [s[0] for s in sizes], [s[1] for s in sizes]
    df["hash"] = hashes
    issues["unreadable"] = len(bad)
    df = df[df["hash"].notna()]

    split_sets = df.groupby("hash")["split"].nunique()
    issues["duplicates_within_data"] = int(df["hash"].duplicated().sum())
    issues["duplicates_across_splits"] = int((split_sets > 1).sum())
    # Keep one copy of each duplicate, in the earliest split, so no test photo is also a training photo
    rank = {"train": 0, "valid": 1, "test": 2}
    df = (df.assign(_r=df["split"].map(rank)).sort_values(["_r", "path"])
            .drop_duplicates("hash", keep="first").drop(columns="_r"))
    issues["duplicates_removed"] = issues["duplicates_within_data"]
    log.info("Validation: %s", issues)
    return df.reset_index(drop=True), issues


# =========================================== 3. explore
def explore(df: pd.DataFrame, issues: dict, cfg: dict, out: Path) -> dict:
    """Class balance, damage counts and sizes, photo sizes, example grid."""
    out.mkdir(parents=True, exist_ok=True)
    order = [c for c in cfg["data"]["class_order"] if c in set(df["label"])]

    counts = pd.crosstab(df["label"], df["split"]).reindex(order)
    counts = counts[[s for s in ["train", "valid", "test"] if s in counts.columns]]
    shares = df["label"].value_counts(normalize=True).reindex(order)
    by_class = df.groupby("label").agg(median_damages=("n_damages", "median"),
                                       median_area=("damaged_area", "median"),
                                       mixed_share=("n_classes_in_photo", lambda s: (s > 1).mean())).reindex(order)
    summary = {
        "n_photos": int(len(df)), "issues": issues,
        "counts_by_split": {s: {c: int(v) for c, v in counts[s].items()} for s in counts.columns},
        "class_share": {c: round(float(v), 4) for c, v in shares.items()},
        "majority_class": str(shares.idxmax()), "majority_share": round(float(shares.max()), 4),
        "by_class": {c: {k: round(float(v), 4) for k, v in r.items()} for c, r in by_class.iterrows()},
        "mixed_severity_share": round(float((df["n_classes_in_photo"] > 1).mean()), 4),
        "median_width": int(df["width"].median()), "median_height": int(df["height"].median()),
        "min_side_px": int(df[["width", "height"]].min(axis=1).min()),
    }
    print("\n=== Class counts by split ===\n", counts.to_string())
    print("\n=== Damage per photo, by class ===\n", by_class.round(3).to_string())
    print(f"\nPhotos with more than one severity: {summary['mixed_severity_share']:.1%}")
    print(f"Median photo size: {summary['median_width']} x {summary['median_height']} px")

    # Chart 1: class balance per split
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, rot=0, color=["#4C78A8", "#9ECAE9", "#F58518"][:len(counts.columns)])
    ax.set_title("Photos per severity class and split"); ax.set_xlabel(""); ax.set_ylabel("photos")
    fig.savefig(out / "class_balance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Chart 2: damaged area by class
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df.loc[df["label"] == c, "damaged_area"] for c in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.set_ylabel("share of photo covered by damage boxes")
    ax.set_title("Damaged area by severity class")
    fig.savefig(out / "damaged_area_by_class.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Chart 3: example photos per class
    n = cfg["evaluation"]["n_examples_per_class"]
    fig, axes = plt.subplots(len(order), n, figsize=(3 * n, 3 * len(order)), squeeze=False)
    for i, c in enumerate(order):
        sample = df[df["label"] == c].sample(min(n, (df["label"] == c).sum()), random_state=0)
        for j in range(n):
            ax = axes[i, j]; ax.axis("off")
            if j < len(sample):
                ax.imshow(Image.open(sample.iloc[j]["path"]).convert("RGB"))
                if j == 0:
                    ax.set_title(c, loc="left", color=COLORS.get(c, "black"), fontweight="bold")
    fig.suptitle("Example photos per class"); fig.tight_layout()
    fig.savefig(out / "examples_per_class.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Exploration saved to %s", out)
    return summary


# =========================================== 4. features
def build_encoder(cfg: dict):
    import torch
    from torchvision import models, transforms
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()          # drop the ImageNet classifier: keep the 512 features
    net.eval()
    size = cfg["features"]["image_size"]
    tf = transforms.Compose([transforms.Resize(int(size * 256 / 224)), transforms.CenterCrop(size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf


def extract_features(paths: list[str], cfg: dict, cache: Path | None = None) -> np.ndarray:
    """Pretrained ResNet18 -> 512 numbers per photo. Cached so re-runs are fast."""
    key = hashlib.md5("|".join(paths).encode()).hexdigest()[:12]
    if cache is not None and cfg["features"]["cache"]:
        f = cache / f"features_{key}.npy"
        if f.exists():
            log.info("Loaded cached features for %s photos", len(paths))
            return np.load(f)
    import torch
    net, tf = build_encoder(cfg)
    bs, feats = cfg["features"]["batch_size"], []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
            feats.append(net(x).numpy())
            log.info("Features: %s/%s photos", min(i + bs, len(paths)), len(paths))
    X = np.vstack(feats)
    if cache is not None and cfg["features"]["cache"]:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"features_{key}.npy", X)
    return X


# =========================================== 5. split
def split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    has_test = (df["split"] == "test").any()
    if has_test:
        train_splits = ["train", "valid"] if cfg["data"]["merge_valid_into_train"] else ["train"]
        test_name = "test"
    else:
        train_splits, test_name = ["train"], "valid"
    train = df[df["split"].isin(train_splits)].reset_index(drop=True)
    test = df[df["split"] == test_name].reset_index(drop=True)
    log.info("Train %s photos (%s) / test %s photos (%s)", len(train), "+".join(train_splits), len(test), test_name)
    return train, test, test_name


# ========================================= 6. tune + train
def tune_and_train(X_tr, y_tr, cfg: dict):
    model = Pipeline([("scale", StandardScaler()),
                      ("logreg", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    t = cfg["tuning"]
    folds = min(t["cv_folds"], pd.Series(y_tr).value_counts().min())
    search = GridSearchCV(model, t["grid"], scoring=t["scoring"], n_jobs=-1, refit=True,
                          cv=StratifiedKFold(folds, shuffle=True, random_state=42))
    search.fit(X_tr, y_tr)
    log.info("Best CV %s = %.3f with %s", t["scoring"], search.best_score_, search.best_params_)
    cv = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return search.best_estimator_, search.best_params_, round(float(search.best_score_), 4), cv


# =========================================== 7. evaluate
def evaluate(model, X_te, y_te, y_tr, cfg: dict, test_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    order = [c for c in cfg["data"]["class_order"] if c in set(y_tr)]
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    y_te, y_tr = np.asarray(y_te), np.asarray(y_tr)
    majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_tr), 1)), y_tr)
    base = majority.predict(np.zeros((len(y_te), 1)))
    rank = {c: i for i, c in enumerate(order)}
    dist = np.array([abs(rank[a] - rank[b]) for a, b in zip(y_te, pred)])
    recalls = recall_score(y_te, pred, labels=order, average=None, zero_division=0)
    m = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "test_split": test_name,
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "recall_by_class": {c: round(float(r), 4) for c, r in zip(order, recalls)},
        "baseline_accuracy_majority_class": round(accuracy_score(y_te, base), 4),
        "baseline_macro_f1_majority_class": round(f1_score(y_te, base, average="macro"), 4),
        "errors_off_by_one_class": round(float((dist == 1).mean()), 4),
        "errors_off_by_two_classes": round(float((dist == 2).mean()), 4),
        "severe_predicted_as_minor": int(((y_te == "severe") & (pred == "minor")).sum()),
        "minor_predicted_as_severe": int(((y_te == "minor") & (pred == "severe")).sum()),
    }
    print("\n" + classification_report(y_te, pred, labels=order, digits=3, zero_division=0))
    log.info("Test accuracy %.3f | macro-F1 %.3f (baseline %.3f / %.3f)", m["accuracy"], m["macro_f1"],
             m["baseline_accuracy_majority_class"], m["baseline_macro_f1_majority_class"])
    return m, pred, proba


# =========================================== 8. save
def save_artifacts(run_dir: Path, model, metrics: dict, cv: pd.DataFrame, test: pd.DataFrame,
                   pred: np.ndarray, proba: np.ndarray, classes, cfg: dict) -> None:
    figs = run_dir / "figures"; figs.mkdir(parents=True, exist_ok=True)
    order = [c for c in cfg["data"]["class_order"] if c in set(test["label"]) | set(pred)]
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cv.sort_values("rank_test_score").to_csv(run_dir / "cv_results.csv", index=False)

    res = test[["path", "label", "n_damages", "damaged_area"]].copy()
    res["predicted"] = pred
    res["confidence"] = proba.max(axis=1).round(3)
    for i, c in enumerate(classes):
        res[f"p_{c}"] = proba[:, i].round(3)
    res.to_csv(run_dir / "test_predictions.csv", index=False)
    errors = res[res["label"] != res["predicted"]].sort_values("confidence", ascending=False)
    errors.to_csv(run_dir / "errors_to_review.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(test["label"], pred, labels=order, normalize="true", values_format=".2f")
    plt.title("Damage severity: confusion matrix (row-normalized)")
    plt.savefig(figs / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

    # Recall by class vs damaged-area: does the model get big damage right more often?
    fig, ax = plt.subplots(figsize=(7, 4))
    r = metrics["recall_by_class"]
    ax.bar(list(r), list(r.values()), color=[COLORS.get(c, "#4C78A8") for c in r])
    ax.axhline(1 / len(r), color="grey", ls="--", label="random guess")
    ax.set_ylim(0, 1); ax.set_ylabel("recall (share of class found)"); ax.legend()
    ax.set_title("Recall by severity class, test set")
    fig.savefig(figs / "recall_by_class.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # The most confident mistakes, to read like errors_to_review.csv
    n = min(cfg["evaluation"]["n_error_examples"], len(errors))
    if n:
        cols = 4; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, e) in zip(axes.ravel(), errors.head(n).iterrows()):
            ax.imshow(Image.open(e["path"]).convert("RGB"))
            ax.set_title(f"true {e['label']} | pred {e['predicted']} ({e['confidence']:.2f})", fontsize=9)
        fig.suptitle("Most confident mistakes"); fig.tight_layout()
        fig.savefig(figs / "error_examples.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    log.info("Saved model, metrics, predictions, errors and figures to %s", run_dir)
