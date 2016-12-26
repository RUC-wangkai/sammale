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

import numpy as np
from sklearn import datasets

from sammale.datasets import make_simple_data0
from sammale.models import LinearPrimaryFunctionModel
from sammale.visualizers import plot_prediction


def run():
    test1()
    # test2()


def test1():
    x, y = make_simple_data0(nb=100)

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
    for y1, y2 in zip(y, model.predict(x)):
        print '{:2}\t{:2}'.format(y1, y2)
    plot_prediction(model, x, y)


if __name__ == '__main__':
    run()
