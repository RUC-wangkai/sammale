# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: objectives.py
@time: 2016/12/18 23:16


"""
import numpy as np


def MSE(y_real, y_pred):
    return np.square(y_real - y_pred).mean()
