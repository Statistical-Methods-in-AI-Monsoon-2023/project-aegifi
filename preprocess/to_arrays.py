import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import cupy as cp
import cupyx.scipy.sparse as csp

df = pd.read_csv('data/filtered_plots_and_genres.csv')

mlb = MultiLabelBinarizer()

# split the data into X and y
X = df['plot']
y = df['genres']

print(X.shape, y.shape)

y = y.str.strip('][').str.split(', ')
# remove quotes from each item in the list
y = y.apply(lambda row: [item.strip().strip("'") for item in row])

# transform the y data
y = mlb.fit_transform(y)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()

print(X.shape, y.shape)
print(type(X))
X = csp.csr_matrix(X)
print(type(X))

y = cp.array(y)

# save as numpy arrays
#scipy.sparse.save_npz('vectorised_data/X.npz', X)
cp.savez('vectorised_data/X_cp.npz', X)
#np.save('vectorised_data/y.npy', y)
cp.save('vectorised_data/y_cp.npy', y)
