# Machine-Learning-Framework-for-Forecasting-Interstate-and-Civil-Conflict-Onset-and-Continuation
In this paper, a new overall machine learning model for forecasting the onset and continuation of conflict events in countries and over time is developed and tested. The validated the data-preparation script against your uploaded CSV. It reproduces 170,366 total records, 102,219 development records, and 68,147 untouched test observations, with 67,275 benign and 872 attack samples in the test set, consistent with the regenerated analysis.
The package follows the useful sequential/modular organization of the Kostas repository—where preprocessing, feature selection, and ML stages are separated into ordered source files—but explicitly distinguishes that repository as an implementation/workflow guide, while the leakage-free Boruta-compatible, OOF-stacking, and TPE-XGBoost procedures belong to your revised methodology. The reference repository itself uses ordered Python/Jupyter stages and asks users of its source code to cite the associated University of Essex thesis. 


Python 3.6 was used to create the application files. Before running the files, it must be ensured that Python 3.6 and the following libraries are installed.
Library 	Task
Sklearn 	Machine Learning Library
Numpy 	Mathematical Operations
Pandas 	Data Analysis Tools
Matplotlib 	Graphics and Visuality

The implementation phase consists of 4 steps, which are: 1- Pre-processing 2- Statistics 3- Feature Selection 4- Machine Learning Implementation

Each of these steps contains one or more Python files. The same file was saved with both "py" and "ipynb" extensions. The code they contain is exactly the same. The file with the ipynb extension has the advantage of saving the state of the last run of that file and the screen output.

Thus, screen output can be seen without re-running the files. Files with the ipynb extension can be run using the jupyter notebook program. When running the codes, the sequence numbers in the filenames should be followed.

Because the output of almost every program is the prerequisite for the operation of the next program. Each step is described in detail below.
1 - Pre-processing

This step consists of a single file (preprocessing.ipynb). For this program to work, the dataset (CIC-IDS2017) files must be in the "CSVs" folder in the same location as the program. The dataset files can be access here . (The reason that these files are given an external link is that the maximum limit of the file in the cseegit system is 10 MB)

As a result of executing this file, a file named "all_data.csv" is created. This file is a prerequisite for the other steps to work.

The most recent runtime of this file was recorded as 328 seconds. The technical specifications of the computer on which it is run are given below.
		
Central Processing Unit 	: 	Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz 2.90 GHz
Random Access Memory 	: 	8 GB (7.74 GB usable)
Operating System 	: 	Windows 10 Pro 64-bit
Graphics Processing Unit 	: 	AMD Readon (TM) 530
2 - Statistics

This step consists of a single file (statistics.ipynb). This program examines the file "all_data.csv" and prints the statistics of attack and benign registry on this screen. It is not a prerequisite for any file. It only gives information.

The last run time of this file was recorded as 13 seconds.

3 - Feature Selection

This step consists of two files.
a - feature_selection_for_attack_files.ipynb

This program uses attack files located under the "attacks" folder. The aim of this program is to determine which features are important for each attack. For this purpose, It is used the Random Forest Regressor algorithm to calculate the importance weights of the features in the dataset. These acquired features are used in machine learning section As a screen output, it sorts its features and weights from large to small and shows them on the bar chart (average 20 attributes per attack type).

The most recent run of this file was recorded as 4817 seconds.
b - feature_selection_for_all_data.ipynb

This program applies the previous step to the entire data set. Thus, it creates the feature importance weights of that is valid for the entire dataset. It uses the "all_data.csv" file and the Random Forest Regressor algorithm. As a screen output, it sorts its features and weights from large to small and shows them on the bar chart (20 attributes in total for all attacks).

The last run time of this file was recorded as 25929 seconds.
4 - Machine Learning Implementation

This step applies the machine learning algorithms to the data set and consists of 5 files.
a - machine_learning_implementation_for_attack_files.ipynb

this program uses the attack files under the "./attacks/" folder as a dataset. The features used are the 4 features with the highest weight for each file, produced by the feature_selection_for_attack_files file. This file applies 7 machine learning algorithms to each file 10 times and prints the results of these operations on the screen and in the file "./attacks/results_1.csv". It also creates box and whisker graphics of the results and prints them both on the screen and in the "./attacks/result_graph_1/" folder.

The last run time of this file was recorded as 3601 seconds.
b - machine_learning_implementation_with_18_feature.ipynb

This program implements machine learning methods in the file "all_data.csv". Uses the features used in the previous step. The set of features to be used consists of combining the 4 features with the highest importance-weight achieved for each attack in "machine_learning_implementation_for_attack_files" step under a single roof. Thus, 4 features are obtained from each of the 12 attack types, resulting in a pool of features consisting of 48 attributes. After the repetitions are removed, the number of features is 18.

This file applies 7 machine learning algorithms to "all_data.csv" file 10 times and prints the results of these operations on the screen and in the file "./attacks/results_2.csv". It also creates box and whisker graphics of the results and prints them both on the screen and in the "./attacks/result_graph_2/" folder.

The last run time of this file was recorded as 25082 seconds.
c - machine_learning_implementation_with_7_feature.ipynb

This program implements machine learning methods in the file "all_data.csv". The features used are the 7 features with the highest weight, produced by the feature_selection_for_all_data file. This file applies 7 machine learning algorithms to "all_data.csv" file 10 times and prints the results of these operations on the screen and in the file "./attacks/results_3.csv". It also creates box and whisker graphics of the results and prints them both on the screen and in the "./attacks/result_graph_3/" folder.

The last run time of this file was recorded as 12714 seconds.
d - ml_f_measure_comparison.ipynb

This program runs with the file "all_data.csv". It finds feature giving the highest f-measure for Naive Bayes, QDA, and MLP algorithms, and prints them on the screen.

The last run time of this file was recorded as 2092 seconds.
e- machine_learning_implementation_final.ipynb

This program uses "all_data.csv" file as dataset. In feature selection, it follows a different path. To improve performance for the Naive Bayes, QDA and MLP algorithms, it uses the features generated by the ml_F-criterion_comparison file. In the other four algorithms, it uses 7 features with the highest significance, generated by the feature_selection_for_all_data file.

This file applies 7 machine learning algorithms to "all_data.csv" file 10 times and prints the results of these operations on the screen and in the file "./attacks/results_final.csv". It also creates box and whisker graphics of the results and prints them both on the screen and in the "./attacks/result_graph_final/" folder.

The last run time of this file was recorded as 18561 seconds.
Citations

inside:
data/raw/
the complete experiment is executed with:
python run_all.py
The complete runtime is automatically stored in:
outputs/logs/end_to_end_runtime.json
This directly addresses the reviewer’s request for the end-to-end training/runtime of the complete stacking pipeline, rather than reporting the runtime of an individual classifier.
The package is ready to upload to GitHub, with only two publication-stage items left for you to fill in: the final GitHub/Zenodo URL and your preferred software license.
Code and Data Availability
The source code and reproducibility materials supporting the findings of this study are available as supplementary materials and will be deposited in a public research repository. The reproducibility package contains the complete leakage-free experimental pipeline, including data preparation, stratified train–test partitioning, fold-specific preprocessing, Boruta-compatible feature selection, training-only undersampling, baseline classifier evaluation, generation of out-of-fold predictions, stacking-ensemble construction, TPE/Optuna hyperparameter optimization of the XGBoost meta-learner, independent-test evaluation, and generation of the tables and figures reported in this study.
The experiments use the CICIDS-2017 Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv dataset. A fixed random seed (random_state = 123) is provided to support reproducibility. The supplied implementation reproduces the 60:40 stratified development/test partition, 10-fold stratified cross-validation, model-selection procedure, optimized stacking architecture, performance metrics, ROC curves, confusion matrices, and Boruta feature-selection outputs reported in the revised manuscript.
The implementation was developed with reference to the publicly available Anomaly Detection in Networks Using Machine Learning CICIDS-2017 implementation by Kahraman Kostas as a methodological and code-organization guide. The present implementation extends this framework with the leakage-controlled Boruta feature-selection procedure, training-fold-only imbalance correction, out-of-fold stacking architecture, and TPE-optimized XGBoost meta-learning procedure required for the experiments reported in this study.
The CICIDS-2017 dataset is publicly available from the Canadian Institute for Cybersecurity. Because of its size and redistribution considerations, the raw dataset is not duplicated in the source-code repository. Instructions identifying the required source CSV and its expected location are provided in the repository README.
