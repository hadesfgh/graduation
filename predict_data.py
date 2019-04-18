import pandas as pd
import tkinter as tk
from tkinter import filedialog            #文件对话框，主要用于打开或者保存文件
import numpy as np


# def PredictData():
#     root = tk.Tk()
#     root.withdraw()                       # 作用是去掉TK框
#     fpath = filedialog.askopenfilename()  # fpath获得文件路径，字符串类型
#     df = pd.read_excel(fpath)
#     value = df.values[1:, :]              # (5040, 17)
#     size0 = len(value)
#     new_unknown_data = {}
#     for i in range(size0):
#         if value[i][1] in new_unknown_data.keys():
#             new_unknown_data[value[i][1]].append(value[i][3:])
#         else:
#             new_unknown_data[value[i][1]] = [value[i][3:]]
#     for key in new_unknown_data.keys():
#         new_unknown_data[key] = np.array(new_unknown_data[key])
#     for key in list(new_unknown_data.keys()):
#         if new_unknown_data[key].shape != (168, 14):
#             del new_unknown_data[key]
#     all_data = []
#     for key in new_unknown_data.keys():
#         all_data.append(new_unknown_data[key].reshape(7, 336))
#     all_data = np.array(all_data)
#     all_data = all_data.reshape(84, 336)
#     return all_data                     # (84, 336)


def PredictData():
    root = tk.Tk()
    root.withdraw()                       # 作用是去掉TK框
    fpath = filedialog.askopenfilename()  # fpath获得文件路径，字符串类型
    df = pd.read_excel(fpath)
    value = df.values[1:, :]              # (5040, 17)
    tem = np.delete(value, 1, 1)
    size0 = len(value)
    new_unknown_data = {}
    for i in range(size0):
        if value[i][1] in new_unknown_data.keys():
            new_unknown_data[value[i][1]].append(tem[i])
        else:
            new_unknown_data[value[i][1]] = [tem[i]]
    for key in new_unknown_data.keys():
        new_unknown_data[key] = np.array(new_unknown_data[key])
    for key in list(new_unknown_data.keys()):
        if new_unknown_data[key].shape != (168, 16):
            del new_unknown_data[key]
    a = np.random.randint(0, 12)
    b = np.random.randint(0, 7)
    c = np.random.randint(0, 24)
    key_ = list(new_unknown_data.keys())[a]
    key_str = str(key_)
    val_ = new_unknown_data[key_]                 # (168, 16)
    data = val_[:, 2:].reshape(7, 336)
    time = val_[:, 1][b*24:(b+1)*24]
    time_str = str(time[c])[:-3]
    name = np.unique(val_[:, 0])
    name_str = str(name)
    return data, b, key_str, time_str, name_str


if __name__ == '__main__':
    # root = tk.Tk()
    # root.withdraw()                       # 作用是去掉TK框
    # fpath = filedialog.askopenfilename()  # fpath获得文件路径，字符串类型
    # df = pd.read_excel(fpath)
    # value = df.values[1:, :]
    # print(value[1][3:])
    # print(type(value[1][3:]))
    a, b, c, d, e = PredictData()
    print(a)
    print(type(a))
    print(b)
    print(c)
    print(d)
    print(e)
