import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from utils import load_data, hit_rate
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix, jaccard_score, classification_report
import pickle
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# load from embeddings
print("Loading data...")
X_train = np.load('embeddings/train_embed.npy', allow_pickle=True)
X_test = np.load('embeddings/test_embed.npy', allow_pickle=True)
_, _, y_train, y_test = load_data()

# multi label prediction
xgb_classifier = xgb.XGBClassifier(verbosity=2, tree_method="hist", n_jobs=39)

# model = MultiOutputClassifier(xgb_classifier)

print("Fitting model...")
st = time.time()
xgb_classifier.fit(X_train, y_train)
train_time =  time.time() - st

st = time.time()
print("Predicting...")

y_train_prob = xgb_classifier.predict_proba(X_train)
y_test_prob = xgb_classifier.predict_proba(X_test)

predict_time = time.time() - st
print("Saving metrics...")
    
thresholds = []
for prob, true_labels in zip(y_train_prob, y_train):
    # errors = np.sum(((prob > prob[:, None]) & (true_labels == 0) | (prob <= prob[:, None]) & (true_labels == 1)) * abs(prob[:, None] - prob), axis=1)
    errors = np.sum(((prob > prob[:, None]) & (true_labels == 0) | (prob <= prob[:, None]) & (true_labels == 1)), axis=1)
    thresholds.append(prob[np.argmin(errors)])

xgb_regressor = xgb.XGBRegressor(verbosity=2, tree_method="hist", n_jobs=39)

xgb_regressor.fit(y_train_prob, thresholds)

y_test_thresh = xgb_regressor.predict(y_test_prob)

predictions = (y_test_prob >= y_test_thresh[:, None]).astype(int)

print(f'Hit Rate: {hit_rate(y_test, predictions)}')
print(f'Jaccard Score: {jaccard_score(y_test, predictions, average="micro")}')


file_name = f'xgb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

file_path = f'metrics/{file_name}'

with open(file_path, 'w') as f:
    f.write(f'Train time: {train_time}\n')
    f.write(f'Predict time: {predict_time}\n')
    f.write(f'Accuracy: {accuracy_score(y_test, predictions)}\n')
    f.write(f'Hamming Score: {1 - hamming_loss(y_test, predictions)}\n')
    f.write(f'Jaccard Score: {jaccard_score(y_test, predictions, average="micro")}\n')
    f.write(f'Hit Rate: {hit_rate(y_test, predictions)}\n')
    f.write('Classification Report:\n')
    f.write(f'{classification_report(y_test, predictions, zero_division=True)}\n')
    # f.write(f'Confusion Matrix: {multilabel_confusion_matrix(y_test, y_pred)}\n')

# # save model
print("saving model..")
xgb_classifier.save_model(f'pretrained/xgb_w2v.json')
print("model saved..")
