import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedKFold, cross_val_score
from itertools import combinations
from sklearn.neural_network import MLPClassifier
  
#-----------------Preprocessing------------------------
#load data
balance_scale = fetch_ucirepo(id=12)

X = balance_scale.data.features.values
y = balance_scale.data.targets.values.ravel()

label_map = {'L': 0, 'B': 1, 'R': 2}
y = np.array([label_map[val] for val in y])


#--------------------------Split Data-------------------------
trainX, testX, trainY, testY = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#Scale numerical features
scaler = StandardScaler()
X_scaled = StandardScaler().fit_transform(X)
trainX_scaled = scaler.fit_transform(trainX)
testX_scaled = scaler.transform(testX) 

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB(),
    "Neural Network": MLPClassifier(max_iter=3000)
}


# 10-fold CV repeated 100 times (1,000 train/eval rounds in total)
rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=42)

results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=rkf,scoring='accuracy', n_jobs=-1)
    results[name] = (scores.mean(), scores.std())
    print(f'{name:20s} mean={scores.mean():.4f} std={scores.std():.4f}')

#-----------------------------Exhaustive Search----------------------------------
feature_names = [f"f{i+1}" for i in range(X.shape[1])]
feature_names[0] = "Left weight"
feature_names[1] = "Left distance"
feature_names[2] = "Right weight"
feature_names[3] = "Right distance"
def exhaustive_search(model, X, y, feature_names, cv):
    best_acc, best_subset = -1, None
    m = X.shape[1]
    for k in range(1, m + 1):
        for combo in combinations(range(m), k):
            scores = cross_val_score(model, X[:, list(combo)], y, cv=cv, scoring='accuracy', n_jobs=-1)
            if scores.mean() > best_acc:
                best_acc = scores.mean()
                best_subset = [feature_names[i] for i in combo]
    return best_subset, best_acc


#-----------------------------------------------------------------------
#-------------------- linear classifier----------------------------------
model = Perceptron(max_iter=1000, random_state=42)
model.fit(trainX_scaled, trainY)

# ----- Predictions -----
train_pred = model.predict(trainX_scaled)
test_pred = model.predict(testX_scaled)

print()
print("==========Linear Classifier==========")
print("=== Predictions ===")
print("Predicted classes:", test_pred)
print()

print("=== Performance ===")
print("Train Accuracy:", accuracy_score(trainY, train_pred))
print("Test Accuracy:", accuracy_score(testY, test_pred))
print()
print("Classification Report:\n",
      classification_report(testY, test_pred, target_names=["Left", "Balanced", "Right"]))

#----exhaustive search linear classifier
best_features, best_accuracy = exhaustive_search(
    model,
    X_scaled,
    y,
    feature_names,
    rkf
)

print()
print("======Linear Classifier======")
print(best_features)
print(best_accuracy)
print()



#--------------------------------------------------------
#----------------logistic regression---------------------

# ----- 3. Train logistic regression -----
logReg = LogisticRegression( solver='lbfgs', max_iter=1000 )
logReg.fit(trainX_scaled, trainY)

# ----- 4. Predictions -----
y_train_pred_regression = logReg.predict(trainX_scaled)
y_test_pred_regression = logReg.predict(testX_scaled)

print()
print("===============Logistic Regression===============")
print("=== Performance ===")

print("Train Accuracy:", round(accuracy_score(trainY, y_train_pred_regression), 3))
print("Test Accuracy:", round(accuracy_score(testY, y_test_pred_regression), 3))
print()

print("Classification Report:\n",
      classification_report(testY, y_test_pred_regression, target_names=["Left", "Balanced", "Right"]))

#------------exhaustive search logistic regression
best_features, best_accuracy = exhaustive_search(
    logReg,
    X_scaled,
    y,
    feature_names,
    rkf
)
print()
print("======Logistic regression======")
print(best_features)
print(best_accuracy)
print()

#-----------------------------------------------------------------------
#-----------------KNN---------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(trainX_scaled, trainY)
#predY = knn.predict(testX_scaled)

Y_train_pred_KNN = knn.predict(trainX_scaled)
y_test_pred_KNN = knn.predict(testX_scaled)

print()
print(y_test_pred_KNN)
print("=======================KNN========================")
print("=== Performance ===")

print("Train Accuracy:", accuracy_score(trainY, Y_train_pred_KNN))
print("Test Accuracy:", accuracy_score(testY, y_test_pred_KNN))
print()
print("Classification Report:\n",
      classification_report(testY, y_test_pred_KNN, target_names=["Left", "Balanced", "Right"]))
#print(y_test_pred_KNN)


best_features, best_accuracy = exhaustive_search(
    knn,
    X_scaled,
    y,
    feature_names,
    rkf
)

print()
print("KNN")
print(best_features)
print(best_accuracy)
print()

#---------------Guaissian Naive Bayes------------------------------
GNB = GaussianNB()

GNB.fit(trainX_scaled, trainY)
pred = GNB.predict(testX_scaled)
acc = accuracy_score(testY, pred)

print()
print("===================GaussianNB=====================")
print("Accuracy:", acc)
#for i in range(100):
#    print(f"Input: {testX[i]}  Predicted: {pred[i]}  True: {testY[i]}")

#-----------exhaustive search GuassianNB
best_features, best_accuracy = exhaustive_search(
    GNB,
    X_scaled,
    y,
    feature_names,
    rkf
)

print()
print("GNB")
print(best_features)
print(best_accuracy)
print()

#-------------------------Neural Network------------------------------
mlp = MLPClassifier(hidden_layer_sizes=(10, 10),
                    max_iter=3000,
                    random_state=42,
                    activation='relu')

mlp.fit(trainX_scaled, trainY)

y_pred = mlp.predict(testX_scaled)
print()
print("====================Neural Network================")
print(f"Model Accuracy: {accuracy_score(testY, y_pred):.4f}")
print("\n[Detailed Classification Report]")
print(classification_report(testY, y_pred, target_names=["Left", "Balanced", "Right"]))
pred_Y = mlp.predict(testX_scaled)
print(pred_Y)

best_features, best_accuracy = exhaustive_search(
    mlp,
    X_scaled,
    y,
    feature_names,
    rkf
)

print()
print("======Nueral Network======")
print(best_features)
print(best_accuracy)
print()







