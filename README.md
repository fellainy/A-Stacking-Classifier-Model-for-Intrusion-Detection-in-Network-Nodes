# CICIDS-2017 Leakage-Free Stacking IDS — Reproducibility Package

This repository reproduces the **revised leakage-free experiments** for the manuscript **“A Stacking Classifier Model for Intrusion Detection in Network Nodes.”** It is organized as a sequential, modular pipeline, following the useful reproducibility style of Kahraman Kostas' public CICIDS-2017 implementation while implementing the present study's own leakage-control, Boruta-compatible feature selection, OOF stacking, and TPE-optimized XGBoost methodology.

## What this package reproduces

- exact CICIDS-2017 Thursday Morning Web Attacks file identification and class counts;
- stratified **60% development / 40% untouched test** partition (`random_state=123`);
- training-fold-only median imputation;
- training-fold-only ~70:30 benign/attack random undersampling;
- training-fold-only Boruta-compatible shadow-feature selection;
- MinMax scaling fit on training folds only;
- 10-fold StratifiedKFold comparison of nine classifiers (**Table 6**);
- selection of five level-0 learners by development-set mean F1;
- generation of **out-of-fold (OOF)** meta-features;
- Optuna/TPE optimization of the XGBoost level-1 learner (**Table 7**);
- stacking CV mean/SD (**Table 8**);
- final training-only Boruta feature table (**Table 9**);
- one-time independent test evaluation (**Table 10**);
- ROC curve, feature-stability figure, test-metric comparison, and stacking confusion matrix;
- end-to-end wall-clock timing of the complete reproducibility pipeline.

## Dataset

Download CICIDS-2017 from the Canadian Institute for Cybersecurity and place the following file here:

```text
data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```



## Folder structure

```text
CICIDS2017_Leakage_Free_Reproducibility_Package/
├── 01_prepare_data.py
├── 02_baseline_cv.py
├── 03_tune_stacking.py
├── 04_final_evaluation.py
├── 05_generate_figures.py
├── run_all.py
├── verify_reference_results.py
├── config.yaml
├── requirements.txt
├── CITATION.cff
├── .gitignore
├── data/
│   ├── raw/
│   └── splits/
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── logs/
│   ├── models/
│   └── reference_results/
└── src/
    ├── boruta_shadow.py
    ├── io_utils.py
    ├── metrics.py
    ├── models.py
    └── preprocessing.py
```

## Environment setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the full experiment

```bash
python run_all.py
```

This runs all five numbered stages in sequence and writes the complete wall-clock runtime to:

```text
outputs/logs/end_to_end_runtime.json
```

This is the runtime that should be reported as the **end-to-end stacking-pipeline reproducibility time**, rather than the training time of one individual classifier.

## Stage-by-stage execution

```bash
python 01_prepare_data.py
python 02_baseline_cv.py
python 03_tune_stacking.py
python 04_final_evaluation.py
python 05_generate_figures.py
```

### Stage 1 — data split
Creates and freezes the stratified train/test indices **before any learned preprocessing or resampling**.

### Stage 2 — leakage-free baseline CV
For every outer fold, imputation, undersampling, shadow-feature selection, and scaling are learned only from fold-training observations. OOF scores are saved for all nine candidate algorithms.

### Stage 3 — OOF stacking + TPE
The five best development-set classifiers are selected by mean F1. Their OOF scores form the five-column level-1 matrix. The XGBoost meta-learner is tuned exclusively on this training-derived matrix.

The configured TPE search space is:

| Parameter | Search space |
|---|---|
| `n_estimators` | integer 50–200 |
| `learning_rate` | log-uniform 0.01–0.20 |
| `max_depth` | integer 2–8 |
| `subsample` | uniform 0.60–1.00 |
| `colsample_bytree` | uniform 0.60–1.00 |
| `reg_alpha` | log-uniform 0.0001–1.00 |
| `reg_lambda` | log-uniform 0.10–5.00 |

The regenerated optimal configuration recorded in the revised results was:

```yaml
n_estimators: 144
learning_rate: 0.05162665105467819
max_depth: 2
subsample: 0.8653944307248673
colsample_bytree: 0.9158406909356689
reg_alpha: 0.00492522233779106
reg_lambda: 2.571793178625206
```

Best training-only TPE objective AUC: **0.999994**.

### Stage 4 — untouched-test evaluation
The pipeline is frozen and refit using development data. The test partition is transformed using training-fitted components and is **never undersampled**. Final metrics and confusion-matrix counts are written to Table 10.

### Stage 5 — figures
Produces publication-ready PNG figures from the rerun outputs.

## Reference regenerated results

The `outputs/reference_results/` directory contains snapshots from the leakage-free rerun used to revise the manuscript. The principal stacking independent-test result was:

- Accuracy: **0.999765** (99.9765%)
- AUC: **0.999970**
- Recall: **0.988532**
- Precision: **0.993088**
- F1: **0.990805**
- Cohen's Kappa: **0.990686**
- MCC: **0.990688**
- Confusion matrix: **TN=67,269; FP=6; FN=10; TP=862**

Run:

```bash
python verify_reference_results.py
```

to compare a new Table 10 with the supplied reference snapshot.

## Boruta-compatible implementation note

The revised run used a transparent Boruta-compatible shadow-feature implementation because the standalone `BorutaPy` package was unavailable in the execution environment. Each iteration:

1. creates a randomly permuted shadow copy of every real feature;
2. fits a Random Forest to real + shadow attributes;
3. records a hit when a real feature's importance exceeds the maximum shadow importance;
4. repeats this for five iterations; and
5. confirms features that exceed the shadow maximum in all five iterations.

Five of five hits corresponds to a one-sided Binomial(5, 0.5) probability of **0.03125**. The selector is fit **inside each training fold only** and again on the complete development set for final fitting.

## Leakage-control statement

The test set is separated at Stage 1 and is never used for:

- imputation parameter estimation;
- feature selection;
- undersampling;
- base-model selection;
- OOF generation;
- TPE hyperparameter optimization; or
- threshold/model design.

This is the central correction made in response to the reviewer.

## Reproducibility notes

Exact bit-for-bit identity can depend on CPU parallelism and library versions, especially Random Forest/Extra Trees/XGBoost. The package fixes all explicit random seeds and stores package bounds in `requirements.txt`. Report the exact environment from the machine used for the archival rerun (e.g., `pip freeze > environment-freeze.txt`).

## Implementation guide acknowledged

The repository structure and general modular workflow were informed by:

Kahraman Kostas, **Anomaly Detection in Networks Using Machine Learning**, University of Essex, 2018. Public implementation: `kahramankostas/Anomaly-Detection-in-Networks-Using-Machine-Learning` on GitHub.

The reference repository separates preprocessing, statistics, attack filtering, feature selection, and machine-learning implementation into ordered scripts/notebooks and asks users of its source code to cite the associated master's thesis. This package does **not** claim that the leakage-free Boruta/OOF/TPE implementation is part of that repository; those components correspond to the revised methodology of the present study.

## Recommended manuscript statement

> **Code and data availability.** The source code and reproducibility materials supporting this study are provided in the accompanying repository. The package includes the complete leakage-free workflow for data partitioning, fold-specific preprocessing, training-only undersampling, Boruta-compatible feature selection, baseline model evaluation, out-of-fold stacking, Optuna/TPE optimization of the XGBoost meta-learner, independent-test evaluation, runtime measurement, and generation of the reported tables and figures. The CICIDS-2017 Thursday Morning Web Attacks CSV is publicly available from the Canadian Institute for Cybersecurity and is not redistributed in the repository. All random seeds, TPE search distributions, final optimized parameters, and execution instructions are recorded in the repository configuration and README.

## License

Add the license chosen by the authors before making the repository public (e.g., MIT, BSD-3-Clause, or another license approved by your institution). Do not copy a license from the implementation-guide repository unless its terms explicitly permit and match your intended use.
