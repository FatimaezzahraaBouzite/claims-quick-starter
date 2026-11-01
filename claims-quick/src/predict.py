"""Try both models on our own inputs.

Run:
  python -m src.predict --text "I rear-ended the car in front when the brakes failed"
  python -m src.predict --text "the brakes failed and I hit the car in front"
  python -m src.predict --photo data/photos/test/images/<some photo>.jpg
"""
import argparse
import joblib

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--photo")
    args = ap.parse_args()

    if args.text:
        model = joblib.load("models/text_crash_model.joblib")
        p = model.predict_proba([args.text.lower()])[0, 1]
        print(f"P(crash described) = {p:.2f}")

    if args.photo:
        from src.image_model import build_encoder, features
        net, tf = build_encoder()
        clf = joblib.load("models/image_severity_model.joblib")
        probs = clf.predict_proba(features(net, tf, [args.photo]))[0]
        print({str(c): round(float(p), 2) for c, p in zip(clf.classes_, probs)})


if __name__ == "__main__":
    main()
