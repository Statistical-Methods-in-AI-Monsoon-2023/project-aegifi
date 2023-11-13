import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from utils import load_data, hit_rate
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

y_prob = model_xgb.predict_proba(X_test)

for index, prob in enumerate(y_prob[:10]):
    errors = []
    for cand_index, cand in enumerate(prob):
        error = 0
        for p in prob:
            if (p >= cand and y_test[index][cand_index] == 0) or (p < cand and y_test[index][cand_index] == 1):
                error += 1
        errors.append(error)
    print(prob[np.argmin(np.array(errors))])


# print(classification_report(y_test, y_pred))
# print(jaccard_score(y_test, y_pred, average='micro'))
# print(hit_rate(y_test, y_pred))