# Sammale: StandAlone Mode MAchine LEarning
 
Sammale is a Simple and Easy-to-use Python ML Library.

Sammale是一个简单的、容易上手的的Python机器学习库.

目前支持算法:

- LinearPrimaryFunctionModel: 基于基函数的线性模型，可用于非线性回归问题与二类分类问题。
基函数支持多项式、三角函数、指数函数、正太函数。训练方法支持普通最小二乘法、加权最小二乘法、随机梯度下降法。
支持L1正则化、L2正则化。

- LinearKernelFunctionModel: 基于核函数的线性模型，可用于非线性回归问题与二类分类问题。
核函数支持高斯核、线性核。训练方法支持普通最小二乘法、加权最小二乘法、随机梯度下降法。
支持L1正则化、L2正则化。

- Support Vector Machine: 支持向量机，可用于二类分类问题。
    - ~~支持SMO(Sequential Minimal Optimization)训练算法。
    《Fast training of support vector machines using sequential minimal optimization》, John C. Platt, 1999.~~
    - 支持将SVM看成Hinge损失的线性模型使用随机梯度下降法训练。
    
- BaggingModel: Ensemble Learning方法之一，基于Bagging学习法。通过对数据集重采样，独立学习多个弱学习器的方法。

- AdaBoost: Ensemble Learning方法之一，基于Boosting学习法。通过不断调整数据集权重，依次学习多个弱学习器的方法。


- LinearRegression: 逻辑斯蒂回归。目前支持y={-1,+1}的二类分类问题。训练方法支持随机梯度下降法。
支持L1正则化、L2正则化。

- Softmax: 待定。

- RandomForest: 待定。

- XGBoost: 待定。

- GDBT: 待定。