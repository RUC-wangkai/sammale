# encoding: utf-8
import numpy as np

from .datasets import resample
from .objectives import MSE, ACC


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

    def fit_SGD(self, x, y, lr=0.1, L1_lambda=0.0, L2_lambda=0.0, nb_epochs=10, log_epoch=1, verbose=True):
        n = x.shape[0]
        for epoch in range(nb_epochs):
            for i in range(n):
                t = self.predict([x[i]])
                err = t - y[i]
                d_theta = self.calculate_primary_function([x[i]])[0] * err
                self.theta -= lr * d_theta + L2_lambda * self.theta + L1_lambda * np.sign(self.theta)
            if epoch % log_epoch == 0 and verbose:
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
    # TODO
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def fit(self):
        pass


class DecisionStump(object):
    def __init__(self):
        self.dimension = 0
        self.threshold = 0
        self.direction = 1

    def predict(self, x):
        return np.sign(self.direction * (x[:, self.dimension] - self.threshold)).reshape(x.shape[0], 1)

    def fit_random(self, x, y):
        """x=nxm, y=nx1, 寻找0-1损失最小的决策数桩"""
        n, m = x.shape
        d = np.random.randint(0, m)
        data = np.hstack((x[:, d].reshape(n, 1), y))
        sorted_data = data[data[:, 0].argsort(axis=0)]

        h = sorted_data[:, 1].cumsum()
        g = sorted_data[::-1, 1].cumsum()
        t = g[-2::-1] - h[:-1]
        ti = np.abs(t).argmax()
        c = np.mean(sorted_data[ti:ti + 2, 0])
        s = np.sign(t[ti])
        self.dimension = d
        self.threshold = c
        self.direction = s

    def fit_random_weighted(self, x, y, w):
        """x=nxm, y=nx1, w=nx1，"""
        n, m = x.shape

        d = np.random.randint(0, m)

        y = y * w  # FIX 如果是y *= t会报错，因为y类型是int32，而w是float32
        data = np.hstack((x[:, d].reshape(n, 1), y))
        sorted_data = data[data[:, 0].argsort(axis=0)]
        h = sorted_data[:, 1].cumsum()
        g = sorted_data[::-1, 1].cumsum()
        t = g[-2::-1] - h[:-1]
        ti = np.abs(t).argmax()
        c = np.mean(sorted_data[ti:ti + 2, 0])
        s = np.sign(t[ti])
        self.dimension = d
        self.threshold = c
        self.direction = s
        weighted_zero_one_loss = 0.5 * (1 - s * t[ti])
        return weighted_zero_one_loss

    def fit_weighted(self, x, y, w):
        """x=nxm, y=nx1, w=nx1，"""
        n, m = x.shape

        min_loss = 2
        for d in range(m):
            y = y * w  # FIXME 如果是y *= t会报错，因为y类型是int32，而w是float32
            data = np.hstack((x[:, d].reshape(n, 1), y))
            sorted_data = data[data[:, 0].argsort(axis=0)]
            h = sorted_data[:, 1].cumsum()
            g = sorted_data[::-1, 1].cumsum()
            t = g[-2::-1] - h[:-1]
            ti = np.abs(t).argmax()
            c = np.mean(sorted_data[ti:ti + 2, 0])
            s = np.sign(t[ti])
            weighted_zero_one_loss = 0.5 * (1 - s * t[ti])
            print 'd:{}, loss:{}, min_loss:{}'.format(d, weighted_zero_one_loss, min_loss)
            if weighted_zero_one_loss < min_loss:
                self.dimension = d
                self.threshold = c
                self.direction = s
                min_loss = weighted_zero_one_loss
        return min_loss


class BaggingModel(object):
    def __init__(self):
        self.stumps = []

    def predict(self, x):
        y = 0
        for stump in self.stumps:
            y += stump.predict(x)
        return np.sign(y / len(self.stumps))

    def fit(self, x, y, nb_bagging, nb_resample, log_epoch=1000):
        self.stumps = []
        for i in range(nb_bagging):
            xx, yy = resample(x, y, nb_resample)
            stump = DecisionStump()
            stump.fit_random(xx, yy)
            self.stumps.append(stump)
            if i % log_epoch == 0:
                print 'i:{}'.format(i)


class AdaBoost(object):
    def __init__(self):
        self.thetas = []
        self.stumps = []

    def predict(self, x):
        y_pred = 0
        n = len(self.thetas)
        for i in range(n):
            y_pred += self.thetas[i] * self.stumps[i].predict(x)
        return np.sign(y_pred)

    def fit(self, x, y, nb_weak_classifier, epsilon=0, log_epoch=1, verbose=True):
        n, m = x.shape
        w = np.ones((n, 1)) / n

        for i in range(nb_weak_classifier):
            # Choose a best weak classifier
            stump = DecisionStump()
            err = stump.fit_random_weighted(x, y, w)
            # print stump.dimension, stump.threshold, stump.direction

            # calculate theta of this classifier
            y_pred = stump.predict(x)
            if np.isclose(err, 0):
                theta = 1
            else:
                # print err
                # print (1 - err) / err
                theta = 0.5 * np.log((1 - err) / err)

            self.stumps.append(stump)
            self.thetas.append(theta)
            # print 'theta:', theta

            # update w
            # factor = np.exp(- y_pred * y)
            factor = np.exp(- theta * y_pred * y)
            w *= factor
            w /= w.sum()

            y_pred = self.predict(x)
            acc = ACC(y_pred, y)
            if i % log_epoch == 0 and verbose:
                print 'i:{:3}, acc:{:4.3}, dim:{:3}, dire:{:4}, thres:{:.4}'.format(i, acc, stump.dimension,
                                                                                    stump.direction,
                                                                                    stump.threshold)
            # print 1 - epsilon, acc, acc >= 1-epsilon
            if acc >= 1 - epsilon:
                break
