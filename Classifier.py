from sklearn import datasets, metrics, model_selection
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



diabetes = datasets.load_breast_cancer()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)




sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



model = MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='sgd', max_iter=100, random_state=0, tol=0.0001)


model.fit(X_train_std, y_train)


y_predicted = model.predict(X_test_std)


print("Multilayer Perceptron Classifier results: ")
print("Recall: %2f" % metrics.recall_score(y_test, y_predicted, average='macro', zero_division=1))
print("Precision Score: %2f" % metrics.precision_score(y_test, y_predicted, average='macro', zero_division=1))
print("Accuracy: %2f" % metrics.accuracy_score(y_test, y_predicted))
print("F1: %2f" % metrics.f1_score(y_test, y_predicted, average='macro'))






