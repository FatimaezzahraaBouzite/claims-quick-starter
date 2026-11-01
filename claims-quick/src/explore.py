"""Explore the NHTSA complaints before modelling. before training anything:
  1. Is the data clean?        (missing values, duplicates, very short texts)
  2. Is the target balanced?   (crash rate overall, by vehicle, by year)
  3. What does the text look like? (length, vocabulary, examples)
  4. Is there leakage?         (does the word "crash" give the answer away?)

Run: python -m src.explore
Output: printed summary + charts in reports/eda/ + reports/eda/eda_summary.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data/complaints.csv")
OUT = Path("reports/eda")
STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "was", "is", "it", "my",
        "i", "that", "this", "with", "had", "have", "be", "as", "when", "while", "were", "but", "not",
        "no", "they", "we", "me", "from", "by", "there", "so", "after", "would", "has", "been", "are",
        "which", "then", "out", "up", "all", "if", "did", "do", "any", "car", "vehicle"}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved reports/eda/{name}")


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", str(text).lower()) if w not in STOP and len(w) > 2]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    summary = {}

    # -------------------------------------------------- 1. quality
    section("1. DATA QUALITY")
    print(f"Rows: {len(df):,}   Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    dup_ids = int(df["odi_number"].duplicated().sum())
    dup_text = int(df["description"].duplicated().sum())
    print(f"\nDuplicate complaint IDs: {dup_ids:,}")
    print(f"Duplicate descriptions (same text, different ID): {dup_text:,}")
    summary.update(n_rows=len(df), duplicate_ids=dup_ids, duplicate_descriptions=dup_text)

    # -------------------------------------------------- 2. target
    section("2. TARGET: CRASH REPORTED?")
    rate = df["crash"].mean()
    print(f"Crash rate: {rate:.1%}  ({df['crash'].sum():,} of {len(df):,})")
    print(f"-> A model that always says 'no crash' is {1 - rate:.1%} accurate. "
          "Accuracy is misleading here; use AUC, precision and recall.")
    print("\nOverlap with the other outcome fields:")
    print(pd.crosstab(df["crash"], df["fire"], margins=True).rename_axis(index="crash", columns="fire"))
    print("\nInjuries reported, by crash flag:")
    print(df.groupby("crash")["injuries"].apply(lambda s: (s > 0).mean()).map("{:.1%}".format).to_string())
    summary["crash_rate"] = round(float(rate), 4)

    by_vehicle = (df.assign(vehicle=df["make"] + " " + df["model"])
                    .groupby("vehicle")["crash"].agg(["count", "mean"]).sort_values("mean"))
    print("\nCrash rate by vehicle:")
    print(by_vehicle.rename(columns={"count": "n", "mean": "crash_rate"}).round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    by_vehicle["mean"].plot.barh(ax=ax, color="#4C78A8")
    ax.axvline(rate, color="grey", ls="--", label=f"overall {rate:.1%}")
    ax.set_xlabel("share of complaints reporting a crash"); ax.legend()
    ax.set_title("Crash rate by vehicle")
    save(fig, "crash_rate_by_vehicle.png")

    by_year = df.groupby("year")["crash"].agg(["count", "mean"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_year.index.astype(str), by_year["count"], color="#BBBBBB", label="complaints")
    ax2 = ax.twinx()
    ax2.plot(by_year.index.astype(str), by_year["mean"], marker="o", color="#E45756", label="crash rate")
    ax.set_xlabel("model year"); ax.set_ylabel("complaints"); ax2.set_ylabel("crash rate")
    ax.set_title("Complaints and crash rate by model year")
    save(fig, "crash_rate_by_year.png")

    print("\nTop 10 components:")
    comp = df["component"].fillna("UNKNOWN").str.split(",").str[0].str.split(":").str[0].str.strip()
    print(df.assign(component_main=comp).groupby("component_main")["crash"]
            .agg(["count", "mean"]).sort_values("count", ascending=False).head(10)
            .rename(columns={"count": "n", "mean": "crash_rate"}).round(3).to_string())

    # ---------------------------------------------- 3. text
    section("3. TEXT")
    n_words = df["description"].fillna("").str.split().str.len()
    df["n_words"] = n_words
    print("Description length (words):")
    print(n_words.describe(percentiles=[0.05, 0.5, 0.95]).round(1).to_string())
    print("\nMedian length by crash flag:")
    print(df.groupby("crash")["n_words"].median().to_string())
    print("-> If crash descriptions are much longer, length alone carries signal.")
    med = df.groupby("crash")["n_words"].median()
    summary.update(median_words=float(n_words.median()),
                   median_words_crash=float(med.get(1, 0)), median_words_no_crash=float(med.get(0, 0)))

    fig, ax = plt.subplots(figsize=(8, 4))
    cap = n_words.quantile(0.99)
    for flag, label, color in [(0, "no crash", "#4C78A8"), (1, "crash", "#E45756")]:
        ax.hist(df.loc[df["crash"] == flag, "n_words"].clip(upper=cap), bins=40, alpha=0.6,
                density=True, label=label, color=color)
    ax.set_xlabel("words per description"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Description length by crash flag")
    save(fig, "length_by_crash.png")

    counts = {flag: Counter(w for t in df.loc[df["crash"] == flag, "description"] for w in set(words(t)))
              for flag in (0, 1)}
    n = {flag: (df["crash"] == flag).sum() for flag in (0, 1)}
    vocab = [w for w in set(counts[0]) | set(counts[1]) if counts[0][w] + counts[1][w] >= 20]
    lift = pd.DataFrame({
        "share_in_crash": [counts[1][w] / n[1] for w in vocab],
        "share_in_no_crash": [counts[0][w] / n[0] for w in vocab],
    }, index=vocab)
    lift["ratio"] = (lift["share_in_crash"] + 1e-4) / (lift["share_in_no_crash"] + 1e-4)
    print("\nWords most typical of CRASH complaints (share of descriptions containing the word):")
    print(lift.sort_values("ratio", ascending=False).head(15).round(3).to_string())
    print("\nWords most typical of NO-CRASH complaints:")
    print(lift.sort_values("ratio").head(15).round(3).to_string())
    as_list = lambda t: [{"word": w, **{k: round(float(v), 4) for k, v in r.items()}} for w, r in t.iterrows()]
    summary["top_words_crash"] = as_list(lift.sort_values("ratio", ascending=False).head(10))
    summary["top_words_no_crash"] = as_list(lift.sort_values("ratio").head(10))

    # -------------------------------------------------- 4. leakage
    section("4. LEAKAGE CHECK")
    summary["leakage"] = {}
    for term in ["crash", "accident", "collision", "airbag"]:
        has = df["description"].str.contains(term, case=False, na=False)
        summary["leakage"][term] = {"crash": round(float(has[df["crash"] == 1].mean()), 4),
                                    "no_crash": round(float(has[df["crash"] == 0].mean()), 4)}
        print(f"'{term}': in {has[df['crash'] == 1].mean():.1%} of crash vs "
              f"{has[df['crash'] == 0].mean():.1%} of no-crash descriptions")
    print("-> Strong words are expected: the model SHOULD use them. It's only leakage if the text\n"
          "   repeats the label mechanically (e.g. a form field pasted in). Read a few examples below.")

    # ------------------------------------------------------------------ 5. examples
    section("5. EXAMPLES TO READ")
    for flag, label in [(1, "CRASH"), (0, "NO CRASH")]:
        print(f"\n--- {label} ---")
        for t in df.loc[df["crash"] == flag, "description"].sample(3, random_state=1):
            print("*", t[:300].replace("\n", " "), "...\n")

    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDone. Charts and summary are in reports/eda/")
    from src.update_readme import main as update_readme
    update_readme()


if __name__ == "__main__":
    main()
