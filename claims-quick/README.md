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
| Crash flag from text | 40,174 NHTSA complaints (8,035 in test) | ROC-AUC 0.981, recall 88.2%, precision 79.9% | always "no crash": 94.8% accuracy |
| Severity from photo | 6,477 photos (248 in test) | accuracy 68.5%, macro-F1 0.653 | most common class: 53.2% accuracy |

**Threshold trade-off (text model):**

| Threshold | % flagged | Precision | Recall |
|---|---|---|---|
| 0.3 | 7.6% | 0.623 | 0.918 |
| 0.4 | 6.4% | 0.723 | 0.896 |
| 0.5 | 5.7% | 0.799 | 0.882 |
| 0.6 | 5.2% | 0.843 | 0.857 |
| 0.7 | 4.8% | 0.879 | 0.826 |

Best settings from cross-validation: `{'logreg__C': 3.0, 'tfidf__max_features': 100000, 'tfidf__ngram_range': [1, 2]}` (CV roc_auc 0.974).

**Words that point to a crash:** crashed, hit, accident, damage, collision, crash, did, police, tree, police report

**Words that point to no crash:** headrest, dealer, cause, transmission, warning, sunroof, reason, dealership, miles, fuel

![text confusion matrix](reports/figures/text_confusion_matrix.png)
![top words](reports/figures/text_top_words.png)

_Auto-generated from `reports/` on 2026-11-01._
<!-- AUTO:results:end -->

## Damage severity from photos

### Data exploration
<!-- AUTO:image_eda:start -->
| Class | train | valid | test | Share | Median damages | Median damaged area |
|---|---|---|---|---|---|---|
| minor | 1,089 | 107 | 41 | 19.1% | 1 | 12.1% |
| moderate | 3,458 | 206 | 132 | 58.6% | 1 | 23.5% |
| severe | 1,274 | 95 | 75 | 22.3% | 1 | 35.1% |

- **6,477 photos** after validation. Majority class: **moderate** (58.6%), so always guessing it would be 58.6% accurate.
- **17.6% of photos contain more than one severity**; each photo takes its most serious damage as its label.
- **Data checks:** 0 photos without labels and 0 unreadable files removed; 2 duplicate files removed, 0 of which appeared in more than one split (kept only in the earliest split, so no test photo is also a training photo).
- **Photo size:** median 640 x 640 px, smallest side 640 px (photos are resized to 224 x 224 for ResNet18).

![class balance](reports/image_eda/class_balance.png)
![damaged area](reports/image_eda/damaged_area_by_class.png)
![examples](reports/image_eda/examples_per_class.png)

_Auto-generated from `reports/image_eda/eda_summary.json` on 2026-11-01._
<!-- AUTO:image_eda:end -->

**My notes on the photo data:** _What we see in the example grid? Are the classes visually distinct?
Is the dataset balanced? Any data issues?_

### Results (held-out test photos)
<!-- AUTO:image_results:start -->
| Metric | Model | Majority-class baseline |
|---|---|---|
| Accuracy | 68.5% | 53.2% |
| Macro-F1 | 0.653 | 0.232 |

- **Test set:** 248 photos (`test` split); trained on 6,229.
- **Recall by class:** minor 56.1% · moderate 68.2% · severe 76.0%.
- **Size of mistakes:** 28.2% of test photos are off by one severity level, 3.2% by two (severe called minor: 6, minor called severe: 2).
- **Best setting from cross-validation:** `{'logreg__C': 0.1}` (CV f1_macro 0.762).

![confusion matrix](reports/figures/image_confusion_matrix.png)
![recall by class](reports/figures/image_recall_by_class.png)
![most confident mistakes](reports/figures/image_error_examples.png)

_Auto-generated from `reports/image_metrics.json` on 2026-11-01._
<!-- AUTO:image_results:end -->

### My analysis of the photo model
<!-- Write here after each run. Questions to answer:
     1. How much better than the baseline is it, and is that enough to be useful?
     2. Which classes get confused? Are mistakes mostly off by one level (minor vs moderate)?
     3. Look at error_examples.png: are the errors the model's fault or ambiguous labels?
     4. Does performance depend on damaged area (open runs/<latest>/test_predictions.csv)?
     5. What would you try next? -->
_Your analysis here._

## Photo pipeline
`python -m src.image_pipeline.run` runs from one config file (`configs/image_severity.yaml`):

load YOLO labels (one label per photo = its most serious damage) → validate (unlabelled and unreadable
photos, duplicate files across splits) → explore (class balance, damaged area, example photos) →
ImageNet-pretrained ResNet18 turns each photo into 512 features (frozen, cached) → standardise +
class-weighted logistic regression, tuned with stratified cross-validation on the training photos →
evaluate once on the test photos (accuracy, macro-F1, recall per class, majority-class baseline,
off-by-one vs off-by-two errors) → save everything in `runs/` → README refreshed.

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
python -m src.image_pipeline.run --stage explore   # photo exploration only -> README
python -m src.image_pipeline.run                   # explore + train + evaluate -> runs/ + README
python -m src.predict --text "the brakes failed and I hit the car in front"
```
