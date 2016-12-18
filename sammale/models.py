# encoding: utf-8
import numpy as np


class LinearPrimaryFunctionModel(object):
    """线性基函数模型"""

    def __init__(self, primary_functions=[]):
        self.primary_functions = primary_functions
        if self.primary_functions:
            self.theta = np.ones_like(self.primary_functions)
        else:
            self.theta = np.array([])

    def add(self, func):
        self.primary_functions.append(func)
        self.theta = np.hstack((self.theta, 1))

    def calculate_primary_function(self, x):
        """x支持0维标量与1维向量"""
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        n = x.shape[0]

        m = len(self.primary_functions)
        p_x = np.empty((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                p_x[i][j] = self.primary_functions[j](x[i])
        return p_x

    def predict(self, x):
        p_x = self.calculate_primary_function(x)
        return np.dot(p_x, self.theta.T).reshape(p_x.shape[0])

    def fit(self, x, y):
        p_x = self.calculate_primary_function(x)
        # 最小二乘法的矩阵求法
        self.theta = np.mat(p_x.T.dot(p_x)).I.dot(p_x.T).dot(y).A