import pandas as pd
import tkinter as tk
from tkinter import filedialog            #文件对话框，主要用于打开或者保存文件
import numpy as np


def TestData():
    root = tk.Tk()
    root.withdraw()                       # 作用是去掉TK框
    fpath = filedialog.askopenfilename()  # fpath获得文件路径，字符串类型
    df = pd.read_excel(fpath)
    value = df.values[1:, :]
    size0 = len(value)
    new_unknown_data = {}
    for i in range(size0):
        if value[i][1] in new_unknown_data.keys():
            new_unknown_data[value[i][1]].append(value[i][3:])
        else:
            new_unknown_data[value[i][1]] = [value[i][3:]]
    for key in new_unknown_data.keys():
        new_unknown_data[key] = np.array(new_unknown_data[key])
    for key in list(new_unknown_data.keys()):
        if new_unknown_data[key].shape != (168, 14):
            del new_unknown_data[key]
    all_data = []
    for key in new_unknown_data.keys():
        all_data.append(new_unknown_data[key].reshape(7, 336))
    all_data = np.array(all_data)
    all_data = all_data.reshape(476, 336)
    return all_data