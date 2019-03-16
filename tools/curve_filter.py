#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: curve_filter.py

import numpy as np
from scipy import optimize

def curve_filter_v1(data, stride=10):
    if isinstance(data, np.ndarray):
        data_length = data.shape[0]
    else:
        raise Exception('请将data转化为np.array格式')

    if data_length <= stride:
        return data

    result = []
    for i in range(data_length-stride):
        temp = data[i:i+stride]
        result.append(np.sum(temp) / stride)
    return np.array(result)










if __name__ == '__main__':
    path0 = '/home/simsimi/下载/top_1_curve.npy'
    path1 = '/home/simsimi/下载/top_1_curve(1).npy'
    path2 = '/home/simsimi/下载/top_1_curve(2).npy'
    path3 = '/home/simsimi/下载/top_1_curve(3).npy'

    top_1_0 = curve_filter_v1(np.load(path0))
    top_1_1 = curve_filter_v1(np.load(path1))
    top_1_2 = curve_filter_v1(np.load(path2))
    top_1_3 = curve_filter_v1(np.load(path3))

    top_1 = np.concatenate([top_1_0, top_1_1, top_1_2, top_1_3], axis=0)


    x = np.arange(top_1.shape[0])
    import matplotlib.pyplot as plt
    plt.plot(x, top_1)
    plt.show()


