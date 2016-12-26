# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from sammale.datasets import make_simple_data1
from sammale.models import BaggingModel
from sammale.models import LinearPrimaryFunctionModel
from sammale.visualizers import plot_prediction


def run():
    test1()
    # test2()


def test1():
    # x, y = datasets.make_moons(100, 0.1)
    # y[y == 0] = -1
    # y = y.reshape((x.shape[0], 1))

    x, y = make_simple_data1(100, 2)

    n, m = x.shape
    model = BaggingModel()
    model.fit(x, y, nb_bagging=10000, nb_resample=8)
    print 'acc:{}'.format(np.sum(model.predict(x) == y) * 1.0 / n)
    plot_prediction(model, x, y, lmbda=0.01)
    plt.show()


def test2():
    x, y = make_simple_data1(100, 4)

    model = LinearPrimaryFunctionModel()
    model.add(lambda t: 1)
    model.add(lambda t: t[0])
    model.add(lambda t: t[1])

    model.fit_SGD(x, y, lr=0.01, L1_lambda=0.0005, nb_epochs=500, log_epoch=10)
    print model.theta
    n = x.shape[0]
    print 'acc:{}'.format(np.sum(np.sign(model.predict(x).reshape(n, 1)) == y) * 1.0 / n)
    plot_prediction(model, x, y)
    plt.show()


if __name__ == '__main__':
    run()
