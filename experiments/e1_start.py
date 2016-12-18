# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e1_start.py
@time: 2016/12/18 10:52


"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sammale import datasets
from sammale import objectives


class LinearPrimaryFunctionModel(object):
    """线性基函数模型"""

    def __init__(self, primary_functions=[]):
        self.primary_functions = primary_functions
        if self.primary_functions:
            self.theta = np.ones_like(self.primary_functions)
        else:
            self.theta = np.array([])

    def add(self, func):
        self.primary_functions.append(func)
        self.theta = np.hstack((self.theta, 1))

    def calculate_primary_function(self, x):
        """x支持0维标量与1维向量"""
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        n = x.shape[0]

        m = len(self.primary_functions)
        p_x = np.empty((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                p_x[i][j] = self.primary_functions[j](x[i])
        return p_x

    def predict(self, x):
        p_x = self.calculate_primary_function(x)
        return np.dot(p_x, self.theta.T).reshape(p_x.shape[0])

    def fit(self, x, y):
        p_x = self.calculate_primary_function(x)
        # 最小二乘法的矩阵求法
        self.theta = np.mat(p_x.T.dot(p_x)).I.dot(p_x.T).dot(y).A


def test1():
    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t)
    model.add(lambda t: np.sin(t * np.pi) / (t * np.pi))

    x, y = datasets.make_simple_curve1(100, 0.0)
    y_pred = model.predict(x)

    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    model.fit(x, y)
    y_pred = model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred)))
    print(model.theta)

    plt.scatter(x, y, c='b', label='y_real')
    plt.scatter(x, y_pred, c='r', label='y_pred')
    plt.legend()
    plt.show()


def test2():
    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t)
    model.add(lambda t: t ** 2)
    model.add(lambda t: t ** 3)
    model.add(lambda t: t ** 4)
    model.add(lambda t: t ** 5)
    model.add(lambda t: t ** 6)
    model.add(lambda t: t ** 7)
    model.add(lambda t: t ** 8)

    x, y = datasets.make_simple_curve1(100, 0.2)
    y_pred = model.predict(x)

    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    model.fit(x, y)
    y_pred = model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    plt.scatter(x, y, c='b', label='y_real')
    plt.scatter(x, y_pred, c='r', label='y_pred')
    plt.legend()
    plt.show()

def run():
    # test1()
    test2()


if __name__ == '__main__':
    run()
