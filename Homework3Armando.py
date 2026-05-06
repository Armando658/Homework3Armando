import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
  

balance_scale = fetch_ucirepo(id=12)

X = balance_scale.data.features.values
y = balance_scale.data.targets.values.ravel()

label_map = {'L': 0, 'B': 1, 'R': 2}
y = np.array([label_map[val] for val in y])


#-------------------- linear classification----------------------------------
trainX, testX, trainY, testY = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

trainX_scaled = scaler.fit_transform(trainX)
testX_scaled = scaler.transform(testX) 
#trainX_scaled = StandardScaler().fit_transform(trainX)
#testX_scaled = StandardScaler().fit_transform(testX)

model = Perceptron(max_iter=1000, random_state=42)
model.fit(trainX, trainY)

# ----- Predictions -----
train_pred = model.predict(trainX)
test_pred = model.predict(testX)

print("=== Predictions ===")
print("Predicted classes:", test_pred)
print()

print("=== Performance ===")
print("Train Accuracy:", accuracy_score(trainY, train_pred))
print("Test Accuracy:", accuracy_score(testY, test_pred))
print()
print("Classification Report:\n",
      classification_report(testY, test_pred, target_names=["Left", "Balanced", "Right"]))
print(test_pred)


#--------------------------------------------------------
#----------------logistic regression---------------------


#X_train, X_test, y_train, y_test = train_test_split(
#    X, y,
#    test_size=0.3,
#    random_state=42,
#    stratify=y
#)

# ----- 3. Train logistic regression -----
model2 = LogisticRegression( solver='lbfgs', max_iter=1000 )
model2.fit(trainX, trainY)

# ----- 4. Predictions -----
y_train_pred_regression = model2.predict(trainX)
y_test_pred_regression = model2.predict(testX)


print("=== Performance ===")
print("Train Accuracy:", round(accuracy_score(trainY, y_train_pred_regression), 3))
print("Test Accuracy:", round(accuracy_score(testY, y_test_pred_regression), 3))
print()

print("Classification Report:\n",
      classification_report(testY, y_test_pred_regression, target_names=["Left", "Balanced", "Right"]))
print(y_test_pred_regression)

#-----------------KNN--------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(trainX_scaled, trainY)
predY = knn.predict(testX_scaled)

Y_train_pred_KNN = knn.predict(trainX_scaled)
y_test_pred_KNN = knn.predict(testX_scaled)

print(y_test_pred_KNN)
print("=== Performance ===")
print("Train Accuracy:", accuracy_score(trainY, Y_train_pred_KNN))
print("Test Accuracy:", accuracy_score(testY, y_test_pred_KNN))
print()
print("Classification Report:\n",
      classification_report(testY, y_test_pred_KNN, target_names=["Left", "Balanced", "Right"]))
print(y_test_pred_KNN)


#---------------Guaissian Naive Bayes------------------------------
model3 = GaussianNB()

model3.fit(trainX, trainY)
pred = model.predict(testX)
acc = accuracy_score(testY, pred)
print("Accuracy:", acc)
for i in range(100):
    print(f"Input: {testX[i]}  Predicted: {pred[i]}  True: {testY[i]}")