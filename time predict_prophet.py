"""
整体的思路是先获得原始数据，把数据变成标准格式
去掉异常值，就是特别大或者特别小的值
再做一阶指数平均，平滑数据，由于绝大部分数据都是变化剧烈，所以系数不能小，0.6比较适合,否则指数平均后数据失真，都0.6了，还有屁的效果
最后使用prophet预测
存在一个问题，数据抖动太剧烈了，预测效果不好
4、6、9、10
"""
from predict_data import PredictData
from fill_normalization import fill, normalization
import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta


def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''

    s_temp = []
    s_temp.append(s[0])
    print(s_temp)
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

time = pd.Series(np.array(c, dtype=float), index=pd.date_range(start='2018-10-08', periods=168, freq='H'))
# print(time)
dict_time = {'ds': time.index, 'y': time.values}
df_time = pd.DataFrame(dict_time)
print(df_time)


# 去掉异常值
td = df_time['y'].describe()  # 描述性统计得到：min，25%，50%，75%，max值
high = td['75%'] + 1.25 * (td['75%'] - td['25%'])  # 定义高点阈值，1.5倍四分位距之外
low = td['25%'] - 1.25 * (td['75%'] - td['25%'])  # 定义低点阈值
forbid_index = df_time['y'][(df_time['y'] > high) | (df_time['y'] < low)].index  # 变化幅度超过阈值的点的索引
print(forbid_index)
li = list(df_time['y'])
for i in range(len(forbid_index)):
        li[forbid_index[i]] = (li[forbid_index[i]-1]+li[forbid_index[i]+1])/2
df_time['y'] = np.array(li)
df_raw = df_time.copy()


# df_time['y'] = np.log(df_time['y'])
# 差分只能把非平稳变得平稳
# df_time['y'] = df_time['y'].diff(1)
# 进行一阶指数平滑
df_time['y'] = exponential_smoothing(0.1, list(df_time['y']))
print(df_time)


# 训练模型并预测
m = Prophet(weekly_seasonality=False)
m.fit(df_time)
future = m.make_future_dataframe(periods=24, freq='H')
# future.tail()
forecast = m.predict(future)
print(forecast)


df_time['y'].plot()
df_raw['y'].plot()
forecast['yhat'].plot()
plt.show()
