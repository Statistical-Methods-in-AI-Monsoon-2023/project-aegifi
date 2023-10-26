# using cuml's Naive Bayes classifier
import cupy as cp
from cuml.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


filtered_df = pd.read_csv('./data/filtered_plots_and_genres.csv')

mlb = MultiLabelBinarizer()

X = filtered_df['plot']
y = filtered_df['genres']

y = mlb.fit_transform(y)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# convert to cupy arrays
X_train = cp.array(X_train)
X_test = cp.array(X_test)
y_train = cp.array(y_train)
y_test = cp.array(y_test)

# create and fit model
model = BernoulliNB()
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)


# save metrics to file
file_name = f'binary_nb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
file_path = f'./metrics/{file_name}'

with open(file_path, 'w') as f:
    f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
    f.write(f'Precision: {precision_score(y_test, y_pred)}\n')
    f.write(f'Recall: {recall_score(y_test, y_pred)}\n')
    f.write(f'F1: {f1_score(y_test, y_pred)}\n')
    f.write(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}\n')