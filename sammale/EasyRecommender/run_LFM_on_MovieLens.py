# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e3_LFM.py
@time: 2016/12/13 20:57


"""

import numpy as np


def regenerate_movielen_dataset(in_file='u.data', out_dir='.', split_index=90000):
    """根据ml-100k数据集中的u.data文件以及给定数量，打乱数据集并生成训练集与测试集，保存到文件中"""
    data = np.loadtxt(in_file, delimiter='\t', dtype=np.int32)
    np.random.shuffle(data)
    train_data, test_data = data[:split_index], data[split_index:]
    print '正在生成训练数据集(shape={})到train.data文件...'.format(train_data.shape)
    np.savetxt('{}/train.data'.format(out_dir), train_data, fmt='%d\t%d\t%d\t%d')
    print '正在生成测试数据集(shape={})到test.data文件...'.format(test_data.shape)
    np.savetxt('{}/test.data'.format(out_dir), test_data, fmt='%d\t%d\t%d\t%d')
    print '生成完毕.'


def load_movielen_dataset(in_dir='.'):
    """加载ml-100k训练集与测试集"""
    print '正在加载数据'
    train_data = np.loadtxt('{}/train.data'.format(in_dir), dtype=np.int32)
    test_data = np.loadtxt('{}/test.data'.format(in_dir), dtype=np.int32)
    return train_data, test_data


def train_LatentFactorModel():
    train_data, test_data = load_movielen_dataset()
    train_data = train_data[:, :3]
    test_data = test_data[:, :3]
    train_data[:, 0:2] -= 1
    test_data[:, 0:2] -= 1

    print 'train_data:{}, test_data:{}'.format(train_data.shape, test_data.shape)
    nb_users, nb_items = 943, 1682

    F = 1000  # 潜在因子数量
    lr = 0.01  # 梯度下降法学习率
    lmbda = 0.0000  # L2正则化系数
    nb_epochs = 100  # 迭代周期
    log_epoch = 1

    # 初始化P矩阵与Q矩阵，LFM模型就是这两个矩阵
    P = np.random.randn(nb_users, F) / np.sqrt(F)
    Q = np.random.randn(F, nb_items) / np.sqrt(F)

    print '开始训练LFM模型'
    for epoch in range(nb_epochs):
        err_sum = 0
        nb_err = 0
        for u_id, i_id, real_r in train_data:
            u_id = int(u_id)
            i_id = int(i_id)
            pred_r = np.dot(P[u_id, :], Q[:, i_id])
            err = real_r - pred_r
            d_Pu = - err * Q[:, i_id]
            d_Qi = - err * P[u_id, :]
            P[u_id, :] += lr * (-d_Pu - lmbda * P[u_id, :])
            Q[:, i_id] += lr * (-d_Qi - lmbda * Q[:, i_id])
            err_sum += np.square(err)
            nb_err += 1
        if epoch % log_epoch == 0:
            print 'epoch:{}, rmse:{}'.format(epoch, err_sum / nb_err)
    print 'final loss:{}'.format(err_sum / nb_err)

    print '开始测试'
    rmse = 0.0
    nb = 0
    for u_id, i_id, real_rating in test_data:
        # 因为ml-100k数据集中用户id与物品id是从1开始计数，所以此处减1
        pred_rating = np.dot(P[int(u_id), :], Q[:, int(i_id)])
        print real_rating, pred_rating
        rmse += np.square(real_rating - pred_rating)
        nb += 1
    rmse = np.sqrt(rmse / nb)
    print '最终mse:{}'.format(rmse)


def run():
    train_LatentFactorModel()


if __name__ == '__main__':
    run()
