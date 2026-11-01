# Auto Claims: Text and Photo Classifiers

Two small, explainable models for car insurance claims intake, built on public data:

1. **NLP:** reads a driver's free-text description of a vehicle problem and flags whether it
   describes a **crash** (TF-IDF + logistic regression).
2. **Computer vision:** classifies **damage severity** (minor / moderate / severe) from a car photo,
   using a pretrained ResNet18 as a feature extractor + logistic regression.


## Data exploration
<!-- AUTO:eda:start -->
| | |
|---|---|
| Complaints | 40,458 |
| Crash rate | 5.1% |
| Accuracy of always predicting "no crash" | 94.9% |
| Median words (crash / no crash) | 103 / 86 |
| Duplicate descriptions | 252 |

- **Imbalance:** only 5.1% of complaints report a crash, so a model that always says "no crash" is already 94.9% accurate. I evaluate with ROC-AUC, precision and recall on the crash class instead.
- **Length:** Crash descriptions are longer (median 103 vs 86 words), so length alone carries some signal.
- **Words typical of crash complaints** (share of crash vs no-crash descriptions): **pole** (4.5% vs 0.0%), **sternum** (1.2% vs 0.0%), **ambulance** (3.0% vs 0.0%), **bruises** (2.0% vs 0.0%), **bruising** (2.0% vs 0.0%).
- **Words typical of non-crash complaints:** **overheating**, **turbo**, **quoted**, **creating**, **timing**.
- **Leakage check:** "crash" in 31.1% of crash vs 2.1% of no-crash; "accident" in 23.5% of crash vs 5.9% of no-crash; "collision" in 16.2% of crash vs 3.8% of no-crash; "airbag" in 23.8% of crash vs 1.5% of no-crash. These are words drivers use to describe what happened, not a copy of the label field.

![crash rate by vehicle](reports/eda/crash_rate_by_vehicle.png)
![length by crash flag](reports/eda/length_by_crash.png)

_Auto-generated from `reports/eda/eda_summary.json` on 2026-11-01._
<!-- AUTO:eda:end -->

**My notes on the exploration:** <>

## Results (held-out test data)
<!-- AUTO:results:start -->
| Model | Data | Result | Simple baseline |
|---|---|---|---|
| Crash flag from text | 40,174 NHTSA complaints (8,035 in test) | ROC-AUC 0.981, recall 88.4%, precision 76.4% | always "no crash": 94.8% accuracy |
| Severity from photo | _run `python -m src.image_model`_ | | |

**Threshold trade-off (text model):**

| Threshold | % flagged | Precision | Recall |
|---|---|---|---|
| 0.3 | 7.9% | 0.604 | 0.923 |
| 0.4 | 6.7% | 0.692 | 0.896 |
| 0.5 | 6.0% | 0.764 | 0.884 |
| 0.6 | 5.4% | 0.822 | 0.867 |
| 0.7 | 5.0% | 0.853 | 0.826 |

Best settings from cross-validation: `{'logreg__C': 3.0, 'tfidf__max_features': 50000, 'tfidf__ngram_range': [1, 2]}` (CV roc_auc 0.974).

**Words that point to a crash:** crashed, hit, accident, collision, damage, crash, police, tree, did, rolled

**Words that point to no crash:** headrest, dealer, cause, transmission, sunroof, warning, highway, fuel, miles, reason

![text confusion matrix](reports/figures/text_confusion_matrix.png)
![top words](reports/figures/text_top_words.png)

_Auto-generated from `reports/` on 2026-11-01._
<!-- AUTO:results:end -->

## Analysis of the text model

### How good is it?
- **The task is imbalanced:** only 5.2% of test complaints (414 of 8,035) report a crash. A model that
  always answers "no crash" is already 94.8% accurate, so accuracy (98.0%) says little here. I judge the model on ranking quality and on precision/recall for the crash class.
- **Ranking is strong:** ROC-AUC is 0.981. A randomly chosen crash complaint gets a higher score than a randomly chosen non-crash complaint 98% of the time.
- **Average precision is 0.90, against 0.05 for random guessing** (the crash rate). This is the more
  honest metric for a rare class, because it ignores the easy-to-classify non-crash majority.
- **At the default threshold (0.5)** the model flags 6.0% of complaints, catches 88.4% of crashes
  (366 of 414) and is right 76.4% of the time when it flags one (366 of 479).
- **No sign of overfitting:** the cross-validated AUC on the training set (0.974) is close to the test
  AUC (0.981), so the model generalises to complaints it has never seen.

### Choosing the threshold
The threshold is a business decision, not a statistical one:

- **0.3:** flags 7.9% of complaints and catches 92.3% of crashes, with more false alarms (precision 60%).
- **0.7:** flags 5.0% and is right 85% of the time, but misses 17% of crashes.

For claims triage, a missed serious claim costs more than an extra review, so I would use **0.3–0.4**:
reviewing about 7–8% of complaints to catch about 90–92% of crashes. In production, I would set this
threshold with the claims team, based on how many files their adjusters can review.

### What the model learned
The strongest words are sensible and easy to explain: **crashed, hit, accident, collision, damage,
crash, police, police report, tree, rolled, ditch, injuries**. Words pointing away from a crash describe mechanical or comfort problems and visits to the dealer: **headrest, dealer, dealership, transmission, sunroof, warning, fuel, screen, 000 miles**. Two less obvious words are worth noting:
- **"did"** pushes towards crash, most likely from the frequent phrase "the air bags **did** not deploy", which appears in crash reports.
- **"accelerated"** points to sudden-acceleration incidents, which often end in a collision.

### Error analysis
I read the 25 most confident errors of each type (`errors_to_review.csv`). This is the most informative part of the project: **many errors come from the labels, not the model.**

**False positives** (the model says crash, the label says no crash; 113 in total):
- **About 10 of 25 clearly describe a collision**, yet are labelled "no crash". Examples: a car that ran
  into a house with the airbag deployed and a passenger injured, a car that rolled 3½ times and was
  totaled, a rear-end collision with a police report. The model is right; the label is wrong. NHTSA
  documents that since May 2021 a blank yes/no field is shown as "N", so a crash the driver did not tick
  becomes "no crash".
- **About 5 describe an airbag deploying without a real collision** (brushing a curb, a pothole). These
  are genuinely ambiguous: is it a crash or not?
- **About 6 are near-misses** ("almost hit 5 cars", "nearly hit a tree"). The wording is very close to
  a crash, and a bag-of-words model can't fully separate "hit" from "almost hit".

**False negatives** (the label says crash, the model says no crash; 48 in total):
- **About 15 of 25 don't describe a crash at all** (a fuel pump, a transmission whine, a seat belt
  buckle). The crash may have been reported only in the form, or ticked by mistake. No text model
  can recover it from these descriptions.
- **About 10 describe a crash, but in less common wording** ("slammed into road railing", "resulting in
  impact", "I was t-boned") **or inside a very long description**, where the crash is one sentence among
  many and its words carry less weight.

**What this means:** the 76% precision is a pessimistic estimate. A share of the "false positives" are
correctly identified crashes with a wrong label. These counts come from the 25 most confident errors of
each type, so they are not a random sample. The next step would be to relabel a random sample of
errors to measure the label error rate properly.


## Text pipeline
`python -m src.pipeline.run` runs these steps from one config file (`configs/text_crash.yaml`):

load → validate & clean (drop short and duplicate texts) → stratified train/test split →
TF-IDF + logistic regression in one scikit-learn `Pipeline` → grid search with 5-fold
stratified cross-validation on the training set only → evaluation on the untouched test set
(ROC-AUC, average precision, precision/recall, threshold table, majority-class baseline) →
top words from the coefficients → everything saved in `runs/<name>_<timestamp>/`
(model, metrics, config copy, log, figures, errors to review) → README results refreshed.

## Why these choices
- **Logistic regression for both:** fast, and every prediction can be explained (word weights for
  text). Same family as the GLMs used in insurance pricing.
- **Pretrained ResNet18, not retrained:** with a few thousand photos, reusing a network trained on
  millions of images works better than training one from scratch, and runs on a laptop CPU.
- **class_weight="balanced":** crashes are a minority of complaints; this stops the model from
  ignoring them.

## What I learned
- <which complaints the text model gets wrong and why (see runs/<latest>/errors_to_review.csv)>
- <which severity classes get confused and why (they're subjective; minor vs moderate overlap)>

## Data
- **Text:** [NHTSA vehicle complaints API](https://api.nhtsa.gov/complaints/complaintsByVehicle)
  (US government, public domain): driver descriptions and the crash yes/no field, for 12 popular
  vehicles, model years 2015-2022.
- **Photos:** Roboflow Universe [car_damage_severity](https://universe.roboflow.com/pob/car_damage_severity-cnfnt)
  (CC0). Photos with several damage boxes take the label of the worst damage.

## Limitations
- NHTSA complaints are US defect reports, not insurance claims, and the crash field is filled in
  by the driver.
- Severity labels are subjective, and the photos are web images, not real claim photos.
- These are prototypes to support people, not to make claim decisions.

## Reproduce
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.get_text_data     # downloads complaints (about 5 minutes)
python -m src.explore           # data exploration -> reports/eda/ + README
python -m src.pipeline.run --config configs/text_crash.yaml   # train + evaluate -> runs/ + README
python -m pytest -q             # tests
python -m src.image_model       # photos unzipped in data/photos/ (a few minutes on CPU) -> README
python -m src.predict --text "the brakes failed and I hit the car in front"
```
