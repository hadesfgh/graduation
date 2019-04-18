from predict_data import PredictData
from fill_normalization import fill, normalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

data = PredictData()                          #未知数据产生聚类中心
data = fill(data, 84, 336, 12)
data = data.reshape(2016, 14)
c = data[:, 1]

np.random.seed(5)
a = np.random.randint(0, 12)
c = c[168*a: 168*(a+1)]

time = pd.Series(np.array(c, dtype=float), index=pd.date_range(start='2016-10-08', periods=168, freq='H'))
print(time)

time.plot()
# plt.title("column 1 data and diff data")
# plt.show()

# # ADF单位根检验判断是不是平稳序列
# t = sm.tsa.stattools.adfuller(time, )
# output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"]
#                       , columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)

# 做一阶差分
time = time.diff(1)
time = time.dropna()
# print(type(time))
time.plot()
# plt.show()

# # 再用ADF单位根检验判断是不是平稳序列
# t = sm.tsa.stattools.adfuller(time)
# output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(time, lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(time, lags=40, ax=ax2)
# plt.show()


# arma = sm.tsa.ARMA(time, (5, 2)).fit()
# print(arma.aic, arma.bic, arma.hqic)
# resid = arma.resid
#
# predict_dta = arma.predict(start='2016-10-14-00', end='2016-10-14-23', dynamic=True,)
# print(predict_dta)
#
# # arma.plot_predict(start='2016-10-14-00', end='2016-10-14-23', dynamic=True,)
# # plt.show()
#
# pred = pd.Series(np.array(list(predict_dta), dtype=float), index=pd.date_range(start='2016-10-14', periods=24, freq='H'))
#
# time.plot()
# pred.plot()
# plt.show()

model = ARIMA(time, order=(5, 1, 1), freq='H').fit()

predict_dta = model.predict(start='2016-10-14-00', end='2016-10-14-23', dynamic=True,)
print(predict_dta)

model.plot_predict(start='2016-10-14-00', end='2016-10-14-23', dynamic=True,)
plt.show()

# pred = model.forecast(10)
# print(pred)
