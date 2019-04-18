import numpy as np


def fill(data_arg, m, n, l):
    for i in range(l):
        for column in range(n):
            list_nan = []
            column_num = 0
            column_sum = 0
            for row in range(7*i, 7*(i+1)):
                if np.isnan(data_arg[row][column]):
                    list_nan.append((row, column))
                else:
                    column_sum += data_arg[row][column]
                    column_num += 1
            if column_num == 0:
                column_ave = 0
            else:
                column_ave = round(column_sum / column_num,2)
                # column_ave = column_sum / column_num
            for (r, c) in list_nan:
                data_arg[r][c] = column_ave
    return data_arg


def normalization(data_arg, m, n):
    def MinMaxNormalization(x, max_num, min_num):
        if max_num == min_num:
            return 0
        else:
            x = (x - min_num) / (max_num - min_num)
            return x
    for j in range(n):
        max_num = np.max(data_arg.T[j][:])
        min_num = np.min(data_arg.T[j][:])
        for p in range(m):
            data_arg.T[j][p] = MinMaxNormalization(data_arg.T[j][p], max_num, min_num)
    return data_arg