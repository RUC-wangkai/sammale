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
import math

def make_simple_curve1(nb=100, noise=0.0):
    """f(x) = sin(pi*x) / (pi*x) + 0.1*x + noise*0.3*randn()"""
    x = np.linspace(-3, 3, nb)
    noi = np.random.randn(nb) * 0.3
    y = np.sin(math.pi * x) / (math.pi * x) + 0.1 * x + noise * noi
    return x, y


def make_simple_curve2(nb=100, noise=0.0):
    x = np.linspace(-10, 10, nb)
    noi = np.random.rand(nb) * 10
    y = np.sin(x) * 5 + 0.1 * x ** 2 + noise * noi
    return x, y
