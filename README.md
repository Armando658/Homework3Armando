# Homework3Armando

# CSCI 3329 — Homework 3 Report
## 1. Dataset
- Balance Scale / Siegler, R. (1976). Balance Scale [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5488X. / number of samples: 625 / number of classes : 4
- Class distribution (table or bar chart)
## 2. Preprocessing
- Loaded dataset using fetch_ucirepo 
- No missing values, no irrelevant columns
- Scaled with Standard Scaler
## 3. Part 2 — Algorithm Comparison
|     Algorithm     | Mean Accuracy | Std |
| Linear Classifier |     0.8615    |0.0388|
|Logistic Regression| 0.8690|0.0419|
|        KNN        | 0.8457|0.0418|
|    Gaussian NB    | 0.9046|0.0353|
|   Neural Network  | 0.9823|0.0164|

## 4. Part 3 — Feature Selection
- Search method and justification
| Algorithm | Best Feature Subset | Mean Accuracy | Std |
|-----------|--------------------|---------------|-----|
## 5. Discussion
- Part 2 vs Part 3 comparison
- Per-algorithm observations
- Limitations and ideas for improvement
## 6. Reproduction
- Python version, key library versions : numpy, pandas, scikit-learn, ucimlrepo
- Run command : Python Homework3Armando.py
