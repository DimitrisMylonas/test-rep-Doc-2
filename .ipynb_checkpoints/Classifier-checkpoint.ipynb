{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26ddefa0-0175-4dc0-9d73-0d39d89a94fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import datasets, metrics, model_selection\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "db7a2567-6bf8-44a6-a1b4-df731e9f7346",
   "metadata": {},
   "outputs": [],
   "source": [
    "diabetes = datasets.load_breast_cancer()\n",
    "X = diabetes.data\n",
    "y = diabetes.target\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01311619-eeb0-4757-a0b7-52f1c315712f",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "sc.fit(X_train)\n",
    "X_train_std = sc.transform(X_train)\n",
    "X_test_std = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5f8c866-d38f-421f-a29b-5785a4d1f067",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = MLPClassifier(hidden_layer_sizes=(10), activation='relu', solver='sgd', max_iter=100, random_state=0, tol=0.0001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbd9d1e1-6a4c-4b47-9a0a-5aa39b0adbf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(X_train_std, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "965c9148-df95-41ff-9eab-2cc9e76c1ed5",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predicted = model.predict(X_test_std)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d5daab07-c046-4b06-9270-8369555b7ca4",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Recall: %2f\" % metrics.recall_score(y_test, y_predicted, average = 'macro', zero_division=1))\n",
    "print(\"Precision Score: %2f\"% metrics.precision_score(y_test, y_predicted, average = 'macro', zero_division=1))\n",
    "print(\"Accuracy: %2f\" % metrics.accuracy_score(y_test, y_predicted))\n",
    "print(\"F1: %2f\" % metrics.f1_score(y_test, y_predicted, average ='macro'))\n",
    "print(\"That's all :)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
