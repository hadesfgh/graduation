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


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''

    s_temp = []
    s_temp.append(s[0])
    for i in range(1, len(s), 1):
        s_temp.append(alpha * s[i-1] + (1 - alpha) * s_temp[i-1])
    return s_temp


data = PredictData()
data = fill(data, 84, 336, 12)
data = data.reshape(2016, 14)
c = data[:, 1]
np.random.seed(5)
a = np.random.randint(0, 12)
c = c[168*a: 168*(a+1)]
# print(c)
# print(type(c))  #numpy.ndarray
# print(c.shape)  # (168,)


time = pd.Series(np.array(c, dtype=float), index=pd.date_range(start='2018-10-08', periods=168, freq='H'))
# print(time)
dict_time = {'ds': time.index, 'y': time.values}
df_time = pd.DataFrame(dict_time)


# 去掉异常值
td = df_time['y'].describe()  # 描述性统计得到：min，25%，50%，75%，max值
high = td['75%'] + 1.5 * (td['75%'] - td['25%'])  # 定义高点阈值，1.5倍四分位距之外
low = td['25%'] - 1.5 * (td['75%'] - td['25%'])  # 定义低点阈值
forbid_index = df_time['y'][(df_time['y'] > high) | (df_time['y'] < low)].index  # 变化幅度超过阈值的点的索引
li = list(df_time['y'])
for i in range(len(forbid_index)):
        li[forbid_index[i]] = (li[forbid_index[i]-1]+li[forbid_index[i]+1])/2
df_time['y'] = np.array(li)
# df_time['y'].plot()


df_time['y'] = exponential_smoothing(0.6, list(df_time['y']))
# df_time['y'].plot()
# plt.show()


raw_seq = df_time['y'].to_list()
# n_steps = 10
n_steps = 20
n_features = 1
X, y = split_sequence(raw_seq, n_steps)
# print(X)
# print(y)


"""
LSTM对输入数据的规模很敏感，特别是当使用sigmoid（默认）或tanh激活功能时。
将数据重新调整到0到1的范围（也称为标准化）可能是一个很好的做法。
我们可以使用scikit-learn库中的MinMaxScaler预处理类轻松地规范数据集。
使用MinMaxScaler().inverse_transform可以恢复数据
"""
scalex = MinMaxScaler(feature_range=(0, 1))
scaley = MinMaxScaler(feature_range=(0, 1))
X = scalex.fit_transform(X)
y = scaley.fit_transform(y.reshape(y.shape[0], 1))


X = X.reshape((X.shape[0], X.shape[1], n_features))
split = int(0.8 * len(X))
x_train = X[:split]
x_test = X[split:]
y_train = y[:split].flatten()
y_test = y[split:].flatten()


# # define model
# model = Sequential()
# model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
# model.add(Dropout(0.1))
# model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(Dropout(0.1))
# model.add(LSTM(100, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # early_stop = EarlyStopping(monitor='loss', patience=10)
# # fit model
# history = model.fit(x_train, y_train, epochs=400, verbose=1, batch_size=10)


model = Sequential()
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# early_stop = EarlyStopping(monitor='loss', patience=10)
# fit model
history = model.fit(x_train, y_train, epochs=800, verbose=1)

model.save("LSTM-.h5")

# plt.plot(history.history['loss'], label='loss')
# plt.title('LOSS')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()
#
# plot_model(model, to_file='model-LSTM.png', show_shapes=True)

model = load_model("LSTM-.h5")

score_lstm = model.evaluate(x_train, y_train)
print('LSTM:', score_lstm)

plt.subplot2grid((1, 2), (0, 0))
pred_train = model.predict(x_train)
inv_y_train = scaley.inverse_transform(y_train.reshape(y_train.shape[0], 1)).flatten()
inv_pred = scaley.inverse_transform(pred_train.reshape(y_train.shape[0], 1)).flatten()
plt.plot(pred_train, label='pred_train')
plt.plot(y_train, label='true_train')
# plt.plot(inv_pred, label='inv_pred')
# plt.plot(inv_y_train, label='inv_y_train')
plt.legend()
# plt.show()


score_lstm = model.evaluate(x_test, y_test)
print('LSTM:', score_lstm)

plt.subplot2grid((1, 2), (0, 1))
pred_test = model.predict(x_test)
plt.plot(pred_test, label='pred_test')
plt.plot(y_test, label='true_test')
plt.legend()
plt.show()


"""
问题1：过拟合严重，暂时解决不了，可能还是需要继续调参，包括时间步和dropout和LSTM层数
问题2：这只是对单个特征做预测,已解决
问题3：最小最大归一化的恢复问题，已解决
"""