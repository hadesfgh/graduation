import numpy as np
import pandas as pd
from predict_data import PredictData
from fill_normalization import fill, normalization
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data = PredictData()
data = fill(data, 84, 336, 12)
np.random.seed(5)
a = np.random.randint(0, 12)
data = data[7*a: 7*(a+1)]
data.resize(168, 14)


scale = MinMaxScaler(feature_range=(0, 1))
data = scale.fit_transform(data)


# choose a number of time steps
n_steps = 10
# convert into input/output
X, y = split_sequences(data, n_steps)
n_features = X.shape[2]


split = int(0.8 * len(X))
x_train = X[:split]
x_test = X[split:]
y_train = y[:split]
y_test = y[split:]


# # define model
# model = Sequential()
# model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# model.add(LSTM(100, activation='relu'))
# model.add(Dense(n_features))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# history = model.fit(x_train, y_train, epochs=400, verbose=1, batch_size=10)


# plt.plot(history.history['loss'], label='loss')
# plt.title('LOSS')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()


# model.save('LSTM-multi.h5')

model = load_model("LSTM-multi.h5")


score_lstm = model.evaluate(x_train, y_train)
print('LSTM:', score_lstm)


# 训练集
plt.subplot2grid((1, 2), (0, 0))
pred = model.predict(x_train)
y_2 = y_train[:, 1]
y_2_pred = pred[:, 1]
plt.plot(y_2_pred, label='pred')
plt.plot(y_2, label='true')
plt.legend()
# plt.show()


score_lstm = model.evaluate(x_test, y_test)
print('LSTM:', score_lstm)


# 测试集
plt.subplot2grid((1, 2), (0, 1))
pred = model.predict(x_test)
y_2 = y_test[:, 1]
y_2_pred = pred[:, 1]
plt.plot(y_2_pred, label='pred')
plt.plot(y_2, label='true')
plt.legend()
plt.show()
