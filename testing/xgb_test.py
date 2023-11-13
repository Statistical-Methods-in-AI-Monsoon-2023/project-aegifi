import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from utils import load_data
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix, classification_report, jaccard_score
# import cupy as cp
# from cupyx.scipy.sparse import spmatrix as cp_csr_matrix
import time
import pickle
import numpy as np

X_test = np.load('embeddings/test_embed.npy', allow_pickle=True)
_, _, _, y_test = load_data()

model_xgb = xgb.XGBClassifier(verbosity=2, tree_method="hist", device="cuda")
model_xgb.load_model('pretrained/xgb_w2v.json')

y_pred = model_xgb.predict(X_test)

print(classification_report(y_test, y_pred))
print(jaccard_score(y_test, y_pred, average='micro'))