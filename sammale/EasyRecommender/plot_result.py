# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: experiments_plot.py
@time: 2016/12/14 16:44

用直方图显示事先跑出的结果。
"""

import matplotlib.pyplot as plt
import numpy as np


def experiment1():
    # 实验1，三种相似度衡量算法下的协同过滤推荐算法的RMSE
    names = ['euclid_similarity', 'pearson_similarity', 'cosine_similarity']
    rmse = np.array([3.62188824141, 1.81571638062, 1.62690662049])

    N = rmse.shape[0]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars

    rects = plt.bar(ind, rmse, width, color='r')
    ax = plt.gca()
    ax.set_title('Experiment1: item-based Collaborative Filtering')
    ax.set_ylabel('Root Mean Squared Error')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(names)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.3f' % height,
                    ha='center', va='bottom')

    autolabel(rects)

    plt.show()


def experiment2():
    # 实验2，不同潜在因子数目的潜在因子模型的RMSE
    names = ['40', '100', '200', '500', '1000']
    rmse = np.array([1.27659541464, 1.21210823458, 1.02579557197, 0.955468048548, 0.930752379247])

    N = rmse.shape[0]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.3  # the width of the bars

    rects = plt.bar(ind, rmse, width, color='b')
    ax = plt.gca()
    ax.set_title('Experiment2: Latent Factor Model')
    ax.set_xlabel('number of latent factors')
    ax.set_ylabel('Root Mean Squared Error')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(names)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.3f' % height,
                    ha='center', va='bottom')

    autolabel(rects)

    plt.show()


def experiment3():
    # 实验3，一起显示实验1与实验2的结果
    names = ['euclid', 'pearson', 'cosine', 'F=40', 'F=100', 'F=200', 'F=500',
             'F=1000']
    rmse = np.array(
        [3.62188824141, 1.81571638062, 1.62690662049, 1.27659541464, 1.21210823458, 1.02579557197, 0.955468048548,
         0.930752379247])

    N = rmse.shape[0]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars

    rects = plt.bar(ind, rmse, width, color='r')
    ax = plt.gca()
    ax.set_title('Conclusion: item-based Collaborative Filtering and Latent Factor Model')
    ax.set_ylabel('Root Mean Squared Error')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(names)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%.3f' % height,
                    ha='center', va='bottom')

    autolabel(rects)

    plt.show()


def run():
    experiment1()
    # experiment2()
    # experiment3()


if __name__ == '__main__':
    run()
