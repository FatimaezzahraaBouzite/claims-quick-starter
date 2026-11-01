"""Fill the auto-generated sections of README.md from the latest results.

README.md contains blocks like:

    <!-- AUTO:eda:start -->
    ...anything here is replaced on every run...
    <!-- AUTO:eda:end -->

Everything OUTSIDE those blocks is yours: write your own comments there and
they are never touched. This runs automatically at the end of src.explore,
src.pipeline.run and src.image_model, or by hand:

    python -m src.update_readme
"""
import json
import re
from datetime import date
from pathlib import Path

README = Path("README.md")
EDA = Path("reports/eda/eda_summary.json")
TEXT = Path("reports/text_metrics.json")
WORDS = Path("reports/text_top_words.json")
IMAGE = Path("reports/metrics.json")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def render_eda(s: dict | None) -> str:
    if not s:
        return "_Run `python -m src.explore` to fill this section._"
    longer = s["median_words_crash"] - s["median_words_no_crash"]
    length_note = (f"Crash descriptions are longer (median {s['median_words_crash']:.0f} vs "
                   f"{s['median_words_no_crash']:.0f} words), so length alone carries some signal."
                   if abs(longer) >= 0.15 * s["median_words_no_crash"] else
                   f"Description length is similar (median {s['median_words_crash']:.0f} vs "
                   f"{s['median_words_no_crash']:.0f} words), so the model must rely on the words themselves.")
    crash_words = ", ".join(f"**{w['word']}** ({_pct(w['share_in_crash'])} vs {_pct(w['share_in_no_crash'])})"
                            for w in s["top_words_crash"][:5])
    safe_words = ", ".join(f"**{w['word']}**" for w in s["top_words_no_crash"][:5])
    leak = "; ".join(f"\"{t}\" in {_pct(v['crash'])} of crash vs {_pct(v['no_crash'])} of no-crash"
                     for t, v in s["leakage"].items())
    return "\n".join([
        f"| | |", "|---|---|",
        f"| Complaints | {s['n_rows']:,} |",
        f"| Crash rate | {_pct(s['crash_rate'])} |",
        f"| Accuracy of always predicting \"no crash\" | {_pct(1 - s['crash_rate'])} |",
        f"| Median words (crash / no crash) | {s['median_words_crash']:.0f} / {s['median_words_no_crash']:.0f} |",
        f"| Duplicate descriptions | {s['duplicate_descriptions']:,} |",
        "",
        f"- **Imbalance:** only {_pct(s['crash_rate'])} of complaints report a crash, so a model that always "
        f"says \"no crash\" is already {_pct(1 - s['crash_rate'])} accurate. I evaluate with ROC-AUC, "
        "precision and recall on the crash class instead.",
        f"- **Length:** {length_note}",
        f"- **Words typical of crash complaints** (share of crash vs no-crash descriptions): {crash_words}.",
        f"- **Words typical of non-crash complaints:** {safe_words}.",
        f"- **Leakage check:** {leak}. These are words drivers use to describe what happened, "
        "not a copy of the label field.",
        "",
        "![crash rate by vehicle](reports/eda/crash_rate_by_vehicle.png)",
        "![length by crash flag](reports/eda/length_by_crash.png)",
        "",
        f"_Auto-generated from `reports/eda/eda_summary.json` on {date.today():%Y-%m-%d}._",
    ])


def render_results(text: dict | None, words: dict | None, image: dict | None) -> str:
    rows = ["| Model | Data | Result | Simple baseline |", "|---|---|---|---|"]
    if text:
        rows.append(f"| Crash flag from text | {text['n_train'] + text['n_test']:,} NHTSA complaints "
                    f"({text['n_test']:,} in test) | ROC-AUC {text['roc_auc']:.3f}, recall "
                    f"{_pct(text['recall'])}, precision {_pct(text['precision'])} | always \"no crash\": "
                    f"{_pct(text['baseline_accuracy_majority_class'])} accuracy |")
    else:
        rows.append("| Crash flag from text | _run `python -m src.pipeline.run`_ | | |")
    img = (image or {}).get("image_severity_classifier")
    if img:
        rows.append(f"| Severity from photo | {img['n_train'] + img['n_test']:,} photos "
                    f"({img['n_test']:,} in {img['test_split']}) | accuracy {_pct(img['accuracy'])}, "
                    f"macro-F1 {img['macro_f1']:.3f} | most common class: "
                    f"{_pct(img['baseline_accuracy_most_common_class'])} accuracy |")
    else:
        rows.append("| Severity from photo | _run `python -m src.image_model`_ | | |")

    out = rows + [""]
    if text:
        out += ["**Threshold trade-off (text model):**", "",
                "| Threshold | % flagged | Precision | Recall |", "|---|---|---|---|"]
        out += [f"| {t['threshold']} | {t['pct_flagged']}% | {t['precision']:.3f} | {t['recall']:.3f} |"
                for t in text["threshold_table"]]
        out += ["", f"Best settings from cross-validation: `{text['best_params']}` "
                    f"(CV {text['cv_scoring']} {text['cv_score']:.3f}).", ""]
    if words:
        out += ["**Words that point to a crash:** " + ", ".join(w for w, _ in words["towards_crash"][:10]),
                "",
                "**Words that point to no crash:** " + ", ".join(w for w, _ in words["towards_no_crash"][:10]),
                ""]
    for path, alt in [("reports/figures/text_confusion_matrix.png", "text confusion matrix"),
                      ("reports/figures/text_top_words.png", "top words"),
                      ("reports/figures/image_confusion_matrix.png", "image confusion matrix")]:
        if Path(path).exists():
            out.append(f"![{alt}]({path})")
    out += ["", f"_Auto-generated from `reports/` on {date.today():%Y-%m-%d}._"]
    return "\n".join(out)


def replace_block(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"(<!-- AUTO:{name}:start -->\n).*?(\n<!-- AUTO:{name}:end -->)", re.S)
    if not pattern.search(text):
        print(f"README has no AUTO:{name} block; skipped")
        return text
    return pattern.sub(lambda m: m.group(1) + content + m.group(2), text)


def main() -> None:
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    text = replace_block(text, "eda", render_eda(_load(EDA)))
    text = replace_block(text, "results", render_results(_load(TEXT), _load(WORDS), _load(IMAGE)))
    README.write_text(text, encoding="utf-8")
    print("README.md auto sections updated")


if __name__ == "__main__":
    main()
