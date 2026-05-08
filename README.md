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
|     Algorithm     | Mean Accuracy |  Std  |
| Linear Classifier |     0.8690    | 0.0419|
|Logistic Regression|     0.8615    | 0.0388|
|        KNN        |     0.8457    | 0.0418|
|    Gaussian NB    |     0.9046    | 0.0353|
|   Neural Network  |     0.9823    | 0.0164|

## 4. Part 3 — Feature Selection
- Nueral Network has the highest accuracy and lowest std, and for feature selection they all worked best with all 4 features
- Using all 4 features makes sense because they are all important to determine the class, if one side has a missing feature, such as a left weight, then the side that has no missing features will have an advantage, having both a distance and a weight against just a weight or distance.
|     Algorithm     |                    Best Feature Subset                 | Mean Accuracy |  Std  |
| Linear Classifier |Left Distance, Left Weight, Right Distance, Right Weight|     0.8692    | 0.0388|
|Logistic Regression|Left Distance, Left Weight, Right Distance, Right Weight|     0.8615    | 0.0419|
|        KNN        |Left Distance, Left Weight, Right Distance, Right Weight|     0.8457    | 0.0418|
|    Gaussian NB    |Left Distance, Left Weight, Right Distance, Right Weight|     0.9202    | 0.0353|
|   Neural Network  |Left Distance, Left Weight, Right Distance, Right Weight|       1.0     | 0.0164|
## 5. Discussion
- Most of the algorithms got more accurate
- For linear classifier, it got slightly more accurate, I believe Linear classifier is limited because it tries to ignore the balance class because it rarely happens, so it mostly guesses left or right, and manages to be accurate because of it, but that also means it isn't reliable
- Logistic classifier has the same problem, but it does try to guess balance, causing it to have a lower accuracy then linear classifier, and doesn't predict balance well
- KNN has the lowest accuracy, but I don't believe it is the worst, I believe that it tries to guess balance and that causes it to be more inaccurate, the problem is when ever the scale is balanced, its nearest neighbors will usually be unbalanced, there isn't really a cluster of balances, it is more randomly spaced
- Guassian Nb performed second best, and I believe this is because when the features look the same, they generally will have the same class. From the beginning I thought this one would do the best because like the tennis example, sunny day + no wind = yes tennis, this one makes sense, heavy left + long distance left vs light right + short distance right will generally mean that the balance will lean left
- Neural network also makes sense to be accurate, although it is a smaller data set, I believe the neural network is still able to find a pattern that can consistently work and lead to a high accuracy
- Limitations and ideas for improvement
## 6. Reproduction
- Python version:  3.14.3
- key library versions : numpy, pandas, scikit-learn, ucimlrepo
- Run command : Python Homework3Armando.py
- random_state = 42
