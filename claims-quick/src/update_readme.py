"""Fill the auto-generated sections of README.md from the latest results.

README.md contains blocks like:

    <!-- AUTO:eda:start -->
    ...anything here is replaced on every run...
    <!-- AUTO:eda:end -->

Everything OUTSIDE those blocks is where I write my own comments. 
This runs automatically at the end of src.explore, src.pipeline.run and src.image_pipeline.run, 
or by hand:
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
IMAGE = Path("reports/image_metrics.json")
IMAGE_EDA = Path("reports/image_eda/eda_summary.json")


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
    img = image
    if img:
        rows.append(f"| Severity from photo | {img['n_train'] + img['n_test']:,} photos "
                    f"({img['n_test']:,} in {img['test_split']}) | accuracy {_pct(img['accuracy'])}, "
                    f"macro-F1 {img['macro_f1']:.3f} | most common class: "
                    f"{_pct(img['baseline_accuracy_majority_class'])} accuracy |")
    else:
        rows.append("| Severity from photo | _run `python -m src.image_pipeline.run`_ | | |")

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
                      ("reports/figures/text_top_words.png", "top words")]:
        if Path(path).exists():
            out.append(f"![{alt}]({path})")
    out += ["", f"_Auto-generated from `reports/` on {date.today():%Y-%m-%d}._"]
    return "\n".join(out)


def render_image_eda(s: dict | None) -> str:
    if not s:
        return "_Run `python -m src.image_pipeline.run --stage explore` to fill this section._"
    classes = list(s["class_share"])
    splits = list(s["counts_by_split"])
    head = "| Class | " + " | ".join(splits) + " | Share | Median damages | Median damaged area |"
    rows = [head, "|---" * (len(splits) + 4) + "|"]
    for c in classes:
        b = s["by_class"][c]
        rows.append(f"| {c} | " + " | ".join(f"{s['counts_by_split'][sp].get(c, 0):,}" for sp in splits)
                    + f" | {_pct(s['class_share'][c])} | {b['median_damages']:.0f} | {_pct(b['median_area'])} |")
    i = s["issues"]
    return "\n".join(rows + [
        "",
        f"- **{s['n_photos']:,} photos** after validation. Majority class: **{s['majority_class']}** "
        f"({_pct(s['majority_share'])}), so always guessing it would be {_pct(s['majority_share'])} accurate.",
        f"- **{_pct(s['mixed_severity_share'])} of photos contain more than one severity**; each photo takes "
        "its most serious damage as its label.",
        f"- **Data checks:** {i['no_label']} photos without labels and {i['unreadable']} unreadable files removed; "
        f"{i['duplicates_within_data']} duplicate files removed, {i['duplicates_across_splits']} of which appeared "
        "in more than one split (kept only in the earliest split, so no test photo is also a training photo).",
        f"- **Photo size:** median {s['median_width']} x {s['median_height']} px, smallest side "
        f"{s['min_side_px']} px (photos are resized to 224 x 224 for ResNet18).",
        "",
        "![class balance](reports/image_eda/class_balance.png)",
        "![damaged area](reports/image_eda/damaged_area_by_class.png)",
        "![examples](reports/image_eda/examples_per_class.png)",
        "",
        f"_Auto-generated from `reports/image_eda/eda_summary.json` on {date.today():%Y-%m-%d}._",
    ])


def render_image_results(m: dict | None) -> str:
    if not m:
        return "_Run `python -m src.image_pipeline.run` to fill this section._"
    rec = " · ".join(f"{c} {_pct(v)}" for c, v in m["recall_by_class"].items())
    out = [
        "| Metric | Model | Majority-class baseline |", "|---|---|---|",
        f"| Accuracy | {_pct(m['accuracy'])} | {_pct(m['baseline_accuracy_majority_class'])} |",
        f"| Macro-F1 | {m['macro_f1']:.3f} | {m['baseline_macro_f1_majority_class']:.3f} |",
        "",
        f"- **Test set:** {m['n_test']:,} photos (`{m['test_split']}` split); trained on {m['n_train']:,}.",
        f"- **Recall by class:** {rec}.",
        f"- **Size of mistakes:** {_pct(m['errors_off_by_one_class'])} of test photos are off by one "
        f"severity level, {_pct(m['errors_off_by_two_classes'])} by two (severe called minor: "
        f"{m['severe_predicted_as_minor']}, minor called severe: {m['minor_predicted_as_severe']}).",
        f"- **Best setting from cross-validation:** `{m['best_params']}` (CV {m['cv_scoring']} {m['cv_score']:.3f}).",
        "",
    ]
    for f, alt in [("image_confusion_matrix.png", "confusion matrix"), ("image_recall_by_class.png", "recall by class"),
                   ("image_error_examples.png", "most confident mistakes")]:
        if Path("reports/figures", f).exists():
            out.append(f"![{alt}](reports/figures/{f})")
    out += ["", f"_Auto-generated from `reports/image_metrics.json` on {date.today():%Y-%m-%d}._"]
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
    text = replace_block(text, "image_eda", render_image_eda(_load(IMAGE_EDA)))
    text = replace_block(text, "image_results", render_image_results(_load(IMAGE)))
    README.write_text(text, encoding="utf-8")
    print("README.md auto sections updated")


if __name__ == "__main__":
    main()
