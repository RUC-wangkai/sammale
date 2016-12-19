# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: kernels.py
@time: 2016/12/19 10:53

核函数。
"""

import numpy as np


def gaussian_kernel(xi, xj, sigma=1):
    """高斯核"""
    return np.exp(-np.sum((xi - xj) ** 2) / (2 * sigma ** 2))
