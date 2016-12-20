# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: datasets.py
@time: 2016/12/18 15:28


"""

import numpy as np


def make_simple_curve0(nb=100, noise=0.0):
    x = np.linspace(-3, 3, nb)
    noi = np.random.randn(nb) * 0.3
    y = x ** 2 + noise * noi
    return x, y


def make_simple_curve1(nb=100, noise=0.0):
    """f(x) = sin(pi*x) / (pi*x) + 0.1*x + noise*0.3*randn()"""
    x = np.linspace(-3, 3, nb)
    noi = np.random.randn(nb) * 0.3
    y = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x + noise * noi
    return x, y


def make_simple_curve2(nb=100):
    x = np.linspace(-3, 3, nb)
    noi = np.random.randn(nb) * 0.3
    y = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x + 0.1 * noi * x
    return x, y


def load_simple_data(nb=100):
    nb_pos = nb / 2
    pos_x = np.random.randn(nb_pos, 2) * 0.2 + 0.8
    pos_y = np.ones(nb_pos, dtype=np.float32)

    nb_neg = nb - nb_pos
    neg_x = np.random.randn(nb_neg, 2) * 0.2 - 0.8
    neg_y = -np.ones(nb_neg, dtype=np.float32)

    return np.vstack((pos_x, neg_x)), np.hstack((pos_y, neg_y))
