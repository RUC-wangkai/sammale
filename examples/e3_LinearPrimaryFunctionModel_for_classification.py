# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e3_LinearPrimaryFunctionModel_for_classification.py
@time: 2016/12/20 21:38


"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sammale.models import LinearPrimaryFunctionModel, LinearKernelFunctionModel
from sammale.kernels import gaussian_kernel
from sammale.visualizers import plot_prediction
from sammale.datasets import load_simple_data


def run():
    # test1()
    test2()


def test1():
    x, y = load_simple_data(nb=100)

    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t[0])
    model.add(lambda t: t[1])

    model.fit_SGD(x, y, lr=0.1, nb_epochs=200, log_epoch=50)
    print model.theta
    plot_prediction(model, x, y)


def test2():
    x, y = datasets.make_moons(100)
    y[y == 0] = -1

    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t[0])
    model.add(lambda t: t[1])
    model.add(lambda t: np.sin(t[0]))
    model.add(lambda t: np.sin(t[1]))
    model.add(lambda t: np.cos(t[0]))
    model.add(lambda t: np.cos(t[1]))

    model.fit_SGD(x, y, lr=0.1, nb_epochs=200, log_epoch=40)
    print model.theta
    plot_prediction(model, x, y)


if __name__ == '__main__':
    run()
