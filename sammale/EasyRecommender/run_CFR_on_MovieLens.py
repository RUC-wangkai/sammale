# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: e1_load.py
@time: 2016/12/10 23:31


"""

import numpy as np

import CFR


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


def convert_data_to_uimat(data, nb_users, nb_items):
    """根据数据集生成user_item矩阵"""
    print '正在转换为user_item矩阵...'
    uimat = np.zeros((nb_users, nb_items), dtype=np.float32)
    for d in data:
        uimat[int(d[0])][int(d[1])] = d[2]
    return uimat


def train_CollaborativeFiltering():
    train_data, test_data = load_movielen_dataset()
    train_data = train_data[:, :3]
    test_data = test_data[:, :3]
    train_data[:, 0:2] -= 1
    test_data[:, 0:2] -= 1

    print 'train_data:{}, test_data:{}'.format(train_data.shape, test_data.shape)
    nb_users, nb_items = 943, 1682

    # 将训练数据转换为user-item评分表
    uimat = convert_data_to_uimat(train_data, nb_users, nb_items)

    # 根据user-item评分表计算相似度矩阵
    simmat = CFR.get_similarity_mat(uimat, mode='col', similarity='euclid_similarity')
    # simmat = CFR.get_similarity_mat(uimat, mode='col', similarity='pearson_similarity')
    # simmat = CFR.get_similarity_mat(uimat, mode='col', similarity='cosine_similarity')

    print '开始测试'
    rmse = 0.0
    nb = 0
    for user_id, item_id, real_rating in test_data:
        # 因为ml-100k数据集中用户id与物品id是从1开始计数，所以此处减1
        pred_rating = CFR.estimate_rating_based_on_item(uimat, simmat, user_id, item_id)
        rmse += np.square(real_rating - pred_rating)
        nb += 1
        if nb % 100 == 0:
            print '当前进度：{}'.format(nb)
    rmse = np.sqrt(rmse / nb)
    print '最终rmse:{}'.format(rmse)


def run():
    train_CollaborativeFiltering()


if __name__ == '__main__':
    run()
