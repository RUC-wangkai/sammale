# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e6_AdaBoost_for_classification.py
@time: 2016/12/26 8:18


"""

import matplotlib.pyplot as plt

from sammale.datasets import make_simple_data1
from sammale.models import AdaBoost
from sammale.visualizers import plot_prediction


def run():
    test1()


def test1():
    x, y = make_simple_data1(300, 2)

    plt.scatter(x[:, 0], x[:, 1], c=y, s=40)
    model = AdaBoost()
    model.fit(x, y, nb_weak_classifier=100, log_epoch=1, epsilon=1e-5)
    plot_prediction(model, x, y)
    plt.show()


if __name__ == '__main__':
    run()
