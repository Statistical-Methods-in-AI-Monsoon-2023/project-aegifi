from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, jaccard_score
from datetime import datetime
import pandas as pd
import scipy
import numpy as np
from utils import load_data, hit_rate
from time import time

X_train, X_test, y_train, y_test = load_data()

model = OneVsRestClassifier(BernoulliNB(alpha=0.5))
train_time = time()
model.fit(X_train, y_train)
train_time = time() - train_time

# predict on test data
predict_time = time()
y_pred = model.predict(X_test)
predict_time = time() - predict_time

for i in range(10):
    print(y_pred[i], y_test[i])

# save metrics to file
file_name = f'binary_nb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
file_path = f'./metrics/{file_name}'

with open(file_path, 'w') as f:
    f.write(f'Train time: {train_time}\n')
    f.write(f'Predict time: {predict_time}\n')
    f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
    f.write(f'Hamming Score: {1 - hamming_loss(y_test, y_pred)}\n')
    f.write(f'Jaccard Score: {jaccard_score(y_test, y_pred, average="micro")}\n')
    f.write(f'Hit Rate: {hit_rate(y_test, y_pred)}\n')
    f.write('Classification Report:\n')
    f.write(f'{classification_report(y_test, y_pred, zero_division=True)}\n')
