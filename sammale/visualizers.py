# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: visualizers.py
@time: 2016/12/20 21:35


"""

import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(model, x, y):
    """绘画出决策边界。只适用于数据维数为2的情况。"""
    real_x_max, real_x_min = np.max(x[:, 0]), np.min(x[:, 0])
    real_y_max, real_y_min = np.max(x[:, 1]), np.min(x[:, 1])
    x_padding = 0.1 * (real_x_max - real_x_min)
    y_padding = 0.1 * (real_y_max - real_y_min)
    x_max = real_x_max + x_padding
    x_min = real_x_min - x_padding
    y_max = real_y_max + y_padding
    y_min = real_y_min - y_padding

    h = 0.008 * (real_x_max - real_x_min)
    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)

    xx, yy = np.meshgrid(xx, yy)
    Z = np.c_[xx.ravel(), yy.ravel()]
    T = np.sign(model.predict(Z))
    T = T.reshape(xx.shape)
    plt.figure('prediction')
    plt.xlim([np.min(xx), np.max(xx)])
    plt.ylim([np.min(yy), np.max(yy)])
    plt.contourf(xx, yy, T, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], s=100, c=y)
    plt.scatter(x[:, 0], x[:, 1], s=30, c=np.sign(model.predict(x)), marker='s')
    plt.show()
