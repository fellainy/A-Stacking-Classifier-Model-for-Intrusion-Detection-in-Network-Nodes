# Source Code and Reproducibility Availability Statement

The source code and reproducibility materials supporting this study are supplied in this repository. The package contains the complete leakage-free experimental workflow for CICIDS-2017 data preparation, stratified train-test partitioning, fold-specific preprocessing, training-only undersampling, Boruta-compatible shadow-feature selection, baseline-model comparison, out-of-fold stacking, Optuna/TPE hyperparameter optimization of the XGBoost meta-learner, independent-test evaluation, end-to-end runtime measurement, and generation of the manuscript tables and figures.

The required source dataset is `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` from CICIDS-2017. The raw dataset is not redistributed in this repository. All principal stochastic operations use the configured random seed (`123`), and the complete TPE search space and final optimized XGBoost parameters are recorded in `config.yaml`.

The modular organization was informed by Kahraman Kostas' public repository *Anomaly Detection in Networks Using Machine Learning* (University of Essex, 2018). That repository is acknowledged as an implementation/workflow guide. The leakage-free Boruta-compatible feature selection, out-of-fold stacking, and TPE-optimized XGBoost procedure in this package implement the methodology of the present revised study.

Repository/DOI: **[INSERT FINAL GITHUB OR ZENODO URL]**
