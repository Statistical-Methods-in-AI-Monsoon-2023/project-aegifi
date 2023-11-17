from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, GRU, Dropout, Conv1D, GlobalAveragePooling1D, GlobalMaxPool1D, concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import torch
import tensorflow as tf
from sklearn.metrics import accuracy_score, jaccard_score, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# MAX_NB_WORDS = 60000
EMBEDDING_DIM = 300

X = torch.load('./data/X.pt').numpy()
y = np.load('./vectorised_data/y.npy')
X = X[:10000]
y = y[:10000]

vocab = torch.load('./data/vocab.pt')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)


# model = Sequential(
#     [
#         Embedding(len(vocab), EMBEDDING_DIM, input_length=X_train.shape[1]),
#         SpatialDropout1D(0.2),
#         GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
#         Dense(20, activation='sigmoid')
#     ]
# )

inp = Input(shape=(X_train.shape[1],))
x = Embedding(len(vocab), EMBEDDING_DIM, input_length=X_train.shape[1])(inp)
x = GRU(128, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
model.summary()

epochs = 2
batch_size = 256

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

predictions = model.predict(X_test)

# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
print(predictions[:10])
predictions = np.round(predictions)
print(predictions[:10])
print(y_test[:10])
print('Jaccard Score: {:.4f}'.format(jaccard_score(y_test, predictions, average='samples')))
print('Classification Report:')
print(classification_report(y_test, predictions, zero_division=True))