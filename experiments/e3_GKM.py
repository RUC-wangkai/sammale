# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e3_GKM.py
@time: 2016/12/19 9:53


"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sammale import datasets
from sammale import objectives
from sammale.kernels import gaussian_kernel
from sammale.models import LinearKernelFunctionModel


def run():
    test1()
    # test2()


def test1():
    """使用最小二乘法训练基于高斯核的核模型。"""
    # x and y are n-d array.
    x, y = datasets.make_simple_curve1(50, 0.1)

    # model = LinearKernelFunctionModel(lambda xi, xj: gaussian_kernel(xi, xj, 0.02))
    # model = LinearKernelFunctionModel(lambda xi, xj: gaussian_kernel(xi, xj, 0.1))
    # model = LinearKernelFunctionModel(lambda xi, xj: gaussian_kernel(xi, xj, 0.15))
    model = LinearKernelFunctionModel(lambda xi, xj: gaussian_kernel(xi, xj, 0.3))
    model.fit_LS(x, y, method='ls')
    y_pred = model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    line_x = np.linspace(-3, 3, 200)
    line_y = model.predict(line_x)

    plt.scatter(x, y, c='b', s=50, marker='s', label='y_real')
    plt.scatter(x, y_pred, c='r', s=20, label='y_pred')
    plt.plot(line_x, line_y, label='bound')
    plt.legend()
    plt.show()


def test2():
    """使用随机梯度下降法训练基于高斯核的核模型与线性核的核模型"""
    x, y = datasets.make_simple_curve1(50, 0.0)

    # 高斯核
    gaussian_kernel_model = LinearKernelFunctionModel(lambda x, kernel: gaussian_kernel(x, kernel, 0.3))
    gaussian_kernel_model.fit_SGD(x, y, lr=0.1, nb_epochs=20, log_epoch=5)
    y_pred1 = gaussian_kernel_model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred1)))

    # 线性核
    linear_kernel_model = LinearKernelFunctionModel(lambda x, kernel: 0.1 * x * kernel)
    linear_kernel_model.fit_SGD(x, y, lr=0.1, nb_epochs=10, log_epoch=1)
    y_pred2 = linear_kernel_model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred2)))

    line_x = np.linspace(-3, 3, 200)
    line_y = gaussian_kernel_model.predict(line_x)

    plt.scatter(x, y, c='b', s=50, marker='s', label='y_real')
    plt.scatter(x, y_pred1, c='r', s=20, label='gaussian kernel')
    plt.scatter(x, y_pred2, c='y', s=20, label='linear kernel')
    plt.plot(line_x, line_y, label='gaussian kernel bound')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
