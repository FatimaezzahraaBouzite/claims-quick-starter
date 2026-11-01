"""Computer vision: classify car damage severity from a photo.

Step 1: a ResNet18 already trained on millions of everyday images (ImageNet)
        turns each photo into 512 numbers that describe it. We do NOT retrain it.
Step 2: a logistic regression learns minor / moderate / severe from those numbers.

Data: a Roboflow car damage severity dataset, exported in YOLO format, unzipped
into data/photos/ (it has train/, valid/, test/ folders and a data.yaml file).
Each photo can have several damage boxes; its label = the worst damage in it.

Runs on a normal laptop CPU (a few minutes).   Run: python -m src.image_model
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torchvision import models, transforms

DATA = Path("data/photos")
REPORTS, FIGS, MODELS = Path("reports"), Path("reports/figures"), Path("models")
ORDER = ["minor", "moderate", "severe"]          # from least to most serious


def photo_labels(split: str, names: list[str]) -> list[tuple[Path, str]]:
    """One label per photo: the most serious damage class among its boxes."""
    rank = {n: next((i for i, o in enumerate(ORDER) if o in n.lower()), -1) for n in names}
    out = []
    for img in sorted((DATA / split / "images").glob("*")):
        lbl = DATA / split / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        classes = [names[int(line.split()[0])] for line in lbl.read_text().splitlines() if line.strip()]
        if classes:
            worst = max(classes, key=lambda c: rank[c])
            out.append((img, ORDER[rank[worst]] if rank[worst] >= 0 else worst))
    return out


def build_encoder():
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()               # drop the ImageNet classifier: keep the 512 features
    net.eval()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf


def features(net, tf, paths: list[Path], batch: int = 32) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            feats.append(net(x).numpy())
            print(f"  {min(i + batch, len(paths))}/{len(paths)} photos", end="\r")
    print()
    return np.vstack(feats)


def main() -> None:
    names = yaml.safe_load(open(DATA / "data.yaml"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    test_split = "test" if (DATA / "test" / "images").exists() else "valid"
    train = photo_labels("train", names)
    test = photo_labels(test_split, names)
    print(f"{len(train)} training photos, {len(test)} {test_split} photos")

    net, tf = build_encoder()
    print("Turning photos into features with the pretrained ResNet18 ...")
    X_tr = features(net, tf, [p for p, _ in train]); y_tr = [l for _, l in train]
    X_te = features(net, tf, [p for p, _ in test]); y_te = [l for _, l in test]

    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(classification_report(y_te, pred, digits=3, zero_division=0))

    labels = [o for o in ORDER if o in set(y_tr)] or sorted(set(y_tr))
    metrics = {"n_train": len(y_tr), "n_test": len(y_te), "test_split": test_split,
               "accuracy": round(float(np.mean(np.array(pred) == np.array(y_te))), 3),
               "macro_f1": round(f1_score(y_te, pred, average="macro"), 3),
               "baseline_accuracy_most_common_class":
                   round(float(np.mean(np.array(y_te) == max(set(y_tr), key=y_tr.count))), 3)}
    print(json.dumps(metrics, indent=2))

    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    path = REPORTS / "metrics.json"
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm["image_severity_classifier"] = metrics
    path.write_text(json.dumps(allm, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=labels, normalize="true", values_format=".2f")
    plt.title("Damage severity from photo: test set (row-normalized)")
    plt.savefig(FIGS / "image_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    joblib.dump(clf, MODELS / "image_severity_model.joblib")


if __name__ == "__main__":
    main()
