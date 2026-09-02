# Reviewer Reproducibility Checklist

- [x] Exact input CSV filename documented.
- [x] Original label counts documented.
- [x] Binary target construction documented.
- [x] Test set separated before any learned operation.
- [x] Fixed split seed (`123`) documented.
- [x] 10-fold stratified CV documented.
- [x] Imputation fitted on fold-training data only.
- [x] Undersampling applied to training data only.
- [x] Boruta-compatible selector implemented and documented.
- [x] Feature-selection stability output saved.
- [x] Nine candidate classifiers specified.
- [x] Five base learners selected only from development CV.
- [x] OOF meta-feature construction implemented.
- [x] XGBoost identified consistently as level-1 learner.
- [x] TPE search distributions stored in `config.yaml`.
- [x] Final optimized XGBoost values stored in `config.yaml`.
- [x] Table 8 mean/SD regenerated from fold-level values.
- [x] Independent test evaluation isolated from Tables 6/8.
- [x] Runtime instrumentation included for complete pipeline.
- [x] Reference outputs from the revised rerun included.
- [x] Figure-generation script included.
- [x] Environment requirements and version snapshot included.
- [x] Implementation-guide repository acknowledged separately from original study code.

Before public release:

- [ ] Add the final GitHub/Zenodo URL to the manuscript.
- [ ] Choose and add an institutional-approved software license.
- [ ] Run `python run_all.py` on the archival machine and commit `outputs/logs/end_to_end_runtime.json`.
- [ ] Save `pip freeze` from that archival machine if exact package locking is required.
