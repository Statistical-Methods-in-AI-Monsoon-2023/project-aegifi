import xgboost as xgb
from utils import load_data
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

X_train, X_test, y_train, y_test = load_data()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print(dtrain, dtest)

xgb_estimator = xgb.XGBClassifier(objective='binary:logistic')

model = MultiOutputClassifier(xgb_estimator)

print("Fitting model...")

model.fit(X_train, y_train)

print("Predicting...")

y_pred = model.predict(X_test)

print("Saving metrics...")

# save metrics to file

file_name = f'xgb_{datetime.now().strftime("%Y%m%d%H%M")}.txt'

file_path = f'./metrics/{file_name}'

with open(file_path, 'w') as f:
    
        f.write(f'Accuracy: {accuracy_score(y_test, y_pred)}\n')
    
        f.write(f'Precision: {precision_score(y_test, y_pred, average="micro")}\n')
    
        f.write(f'Recall: {recall_score(y_test, y_pred, average="micro")}\n')
    
        f.write(f'F1: {f1_score(y_test, y_pred, average="micro")}\n')
    
        f.write(f'Confusion Matrix: {confusion_matrix(y_test, y_pred)}\n')
