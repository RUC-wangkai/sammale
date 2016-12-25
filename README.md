# Sammale: StandAlone Mode MAchine LEarning
 
Sammale is a Simple and Easy-to-use Python ML Library.

Sammale是一个简单的、容易上手的的Python机器学习库.

目前支持算法:

- LinearPrimaryFunctionModel: 基于基函数的线性模型，可用于非线性回归问题与二类分类问题。
基函数支持多项式、三角函数、指数函数、正太函数。训练方法支持普通最小二乘法、加权最小二乘法、随机梯度下降法。

- LinearKernelFunctionModel: 基于核函数的线性模型，可用于非线性回归问题与二类分类问题。
核函数支持高斯核、线性核。训练方法支持普通最小二乘法、加权最小二乘法、随机梯度下降法。

- Support Vector Machine: 支持向量机，可用于二类分类问题。
    - 支持SMO(Sequential Minimal Optimization)训练算法。
    《Fast training of support vector machines using sequential minimal optimization》, John C. Platt, 1999.
    - 支持将SVM看成Hinge损失的线性模型使用随机梯度下降法训练。