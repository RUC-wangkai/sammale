# encoding: utf-8
import numpy as np

from .objectives import MSE


class LinearPrimaryFunctionModel(object):
    """基于基函数的线性模型。

    基函数可以是多项式、三角函数、指数函数、样条函数。

    """

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

    def fit_LS(self, x, y, method='ls'):
        """支持最小二乘法(ls)与加权最小二乘法(wls)求解。"""
        if method not in {'ls', 'wls'}:
            raise Exception('Invalid method:{}, only support "ls" and "wls"!'.format(method))
        p_x = self.calculate_primary_function(x)
        if method == 'ls':
            W = np.eye(len(y))
        elif method == 'wls':
            # TODO 确认一下，加权最小二乘法中的权重矩阵是否这样取值！
            W = np.diag(1 / (np.ones_like(y) * np.var(y)))
        self.theta = np.mat(p_x.T.dot(W).dot(p_x)).I.dot(p_x.T).dot(W).dot(y).A

    def fit_SGD(self, x, y, lr=0.1, L1_lambda=0.0, L2_lambda=0.0, nb_epochs=10, log_epoch=1):
        n = x.shape[0]
        for epoch in range(nb_epochs):
            for i in range(n):
                t = self.predict([x[i]])
                err = t - y[i]
                d_theta = self.calculate_primary_function([x[i]])[0] * err
                self.theta -= lr * d_theta + L2_lambda * self.theta + L1_lambda * np.sign(self.theta)
            if epoch % log_epoch == 0:
                y_pred = self.predict(x)
                print('epoch:{}, mse:{}'.format(epoch, MSE(y, y_pred)))


class LinearKernelFunctionModel(object):
    """基于核函数的线性模型。

    核函数可以是线性核、高斯核。

    """

    def __init__(self, kernel_function):
        self.kernel_function = kernel_function
        self.kernels = None
        self.theta = None

    def calculate_kernel_function(self, x):
        """x仅支持1维向量"""
        x = np.asarray(x)
        n = x.shape[0]
        m = self.kernels.shape[0]
        p_x = np.empty((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                p_x[i][j] = self.kernel_function(x[i], self.kernels[j])
        return p_x

    def predict(self, x):
        if self.theta is None:
            self.theta = np.zeros(self.kernels.shape[0])
        k_x = self.calculate_kernel_function(x)
        return np.dot(k_x, self.theta.T).reshape(k_x.shape[0])

    def fit_LS(self, x, y, method='ls'):
        """支持最小二乘法(ls)与加权最小二乘法(wls)求解。"""
        if method not in {'ls', 'wls'}:
            raise Exception('Invalid method:{}, only support "ls" and "wls"!'.format(method))
        self.kernels = np.array(x)
        p_x = self.calculate_kernel_function(x)
        W = np.eye(len(y))
        if method == 'ls':
            W = np.eye(len(y))
        elif method == 'wls':
            # TODO 确认一下，加权最小二乘法中的权重矩阵是否这样取值！
            W = np.diag(1 / (np.ones_like(y) * np.var(y)))
        self.theta = np.mat(p_x.T.dot(W).dot(p_x)).I.dot(p_x.T).dot(W).dot(y).A

    def fit_SGD(self, x, y, lr=0.1, L1_lambda=0.0, L2_lambda=0.0, nb_epochs=10, log_epoch=1):
        self.kernels = x
        n = x.shape[0]
        self.theta = np.ones(n)
        for epoch in range(nb_epochs):
            for i in range(n):
                t = self.predict([x[i]])
                err = t - y[i]
                d_theta = self.calculate_kernel_function([x[i]])[0] * err
                self.theta -= lr * d_theta + L2_lambda * self.theta + L1_lambda * np.sign(self.theta)
            if epoch % log_epoch == 0:
                y_pred = self.predict(x)
                print('epoch:{}, mse:{}'.format(epoch, MSE(y, y_pred)))


class SVM(object):
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def fit(self):
        pass
