# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e2_WLS.py
@time: 2016/12/19 8:43


"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from sammale import datasets
from sammale import objectives
from sammale.models import LinearPrimaryFunctionModel


def run():
    test1()
    # test2()


def test1():
    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t)
    model.add(lambda t: np.sin(t * np.pi) / (t * np.pi))

    # x, y = datasets.make_simple_curve1(100, 0.9)
    x, y = datasets.make_simple_curve2(100)
    y_pred = model.predict(x)

    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    model.fit(x, y, method='ls')
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
    model.add(lambda t: t ** 9)
    model.add(lambda t: t ** 10)

    x, y = datasets.make_simple_curve1(100, 0.)
    y_pred = model.predict(x)

    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    model.fit(x, y, 'wls')
    y_pred = model.predict(x)
    print('mse:{}'.format(objectives.MSE(y, y_pred)))

    plt.scatter(x, y, c='b', label='y_real')
    plt.scatter(x, y_pred, c='r', label='y_pred')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
