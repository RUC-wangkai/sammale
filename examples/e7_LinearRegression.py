# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e7_LinearRegression.py
@time: 2016/12/29 16:16


"""
import matplotlib.pyplot as plt

from sammale.datasets import make_simple_data1
from sammale.models import LinearRegression
from sammale.visualizers import plot_prediction


def run():
    test1()


def test1():
    x, y = make_simple_data1(10, 2)
    y[y == -1] = 0
    print y

    plt.scatter(x[:, 0], x[:, 1], c=y, s=40)

    model = LinearRegression()
    model.fit_SGD3(x, y, lr=0.1, L1_lambda=0.0, L2_lambda=0.0, nb_epochs=100, log_epoch=10)
    print model.predict(x)
    plot_prediction(model, x, y)
    plt.show()


if __name__ == '__main__':
    run()
