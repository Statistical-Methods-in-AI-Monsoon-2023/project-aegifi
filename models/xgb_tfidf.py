import xgboost as xgb
from utils import load_data
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix
# import cupy as cp
# from cupyx.scipy.sparse import spmatrix as cp_csr_matrix
import time
import pickle

X_train, X_test, y_train, y_test = load_data()

#X_train = cp_csr_matrix(X_train)
#y_train = cp.array(y_train)
#X_test = cp_csr_matrix(X_test)
#y_test_og = y_test
#y_test = cp.array(y_test)

#X_train = X_train[:20000]
#y_train = y_train[:20000]
#X_test = X_test[:2000]
#y_test = y_test[:2000]


xgb_estimator = xgb.XGBClassifier(verbosity=2, tree_method="hist", device="cuda")

# model = MultiOutputClassifier(xgb_estimator)

print("Fitting model...")
st = time.time()
xgb_estimator.fit(X_train, y_train)
train_time =  time.time() - st
st = time.time()
print("Predicting...")

y_pred = xgb_estimator.predict(X_test)
predict_time = time.time() - st
print("Saving metrics...")

# save metrics to file

for i in range(5):
    print(y_pred[i], y_test[i])

file_name = f'xgb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

file_path = f'metrics/{file_name}'

with open(file_path, 'w') as f:
    f.write(f'Train time: {train_time}\n')
    f.write(f'Predict time: {predict_time}\n')
    f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
    f.write(f'Hamming Score: {1 - hamming_loss(y_test, y_pred)}\n')
    f.write(f'Precision: {precision_score(y_test, y_pred, average="micro", zero_division=1)}\n')
    f.write(f'Recall: {recall_score(y_test, y_pred, average="micro", zero_division=1)}\n')
    f.write(f'F1: {f1_score(y_test, y_pred, average="micro", zero_division=1)}\n')
    f.write(f'Confusion Matrix: {multilabel_confusion_matrix(y_test, y_pred)}\n')

# save model
print("saving model..")
pickle.dump(xgb_estimator, open("pretrained/xgb_tfidf.pkl", "wb"))
print("model saved..")
