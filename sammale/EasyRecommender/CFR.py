# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: CFR.py
@time: 2016/12/10 23:29

Collaborative Filtering Recommendation，协同过滤算法
"""

import numpy as np
import numpy.linalg as la


def load_simple_data():
    return np.array([[0, 0, 0, 2, 2],
                     [0, 0, 0, 3, 3],
                     [0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0],
                     [2, 2, 2, 0, 0],
                     [5, 5, 5, 0, 0],
                     [1, 1, 1, 0, 0]], dtype=np.float32)


def load_simple_data2():
    return np.array([[4, 4, 0, 2, 2],
                     [4, 0, 0, 3, 3],
                     [4, 0, 0, 1, 1],
                     [1, 1, 1, 2, 0],
                     [2, 2, 2, 0, 0],
                     [5, 5, 5, 0, 0],
                     [1, 1, 1, 0, 0]], dtype=np.float32)


def load_simple_data3():
    return np.array([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                     [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                     [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                     [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                     [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                     [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                     [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                     [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                     [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                     [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                     [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]], dtype=np.float32)


def euclid_similarity(u, v):
    """基于欧式距离的相似度"""
    d = la.norm(u - v)
    return 1. / (1. + d)


def pearson_similarity(u, v):
    """皮尔逊相关系数"""
    if len(u) < 3:
        return 1
    if len(np.nonzero(np.logical_and(u, v))[0]) == 0:
        return 0
    else:
        return 0.5 * (1. + np.corrcoef(u, v)[0][1])


def cosine_similarity(u, v):
    """余弦相似度"""
    norm_u = la.norm(u)
    norm_v = la.norm(v)
    if len(np.nonzero(np.logical_and(u, v))[0]) == 0 or norm_u == 0 or norm_v == 0:
        c = -1
    else:
        c = np.dot(u, v) / (norm_u * norm_v)
    return 0.5 * (1. + c)


_sim_dict = {
    'euclid_similarity': euclid_similarity,
    'pearson_similarity': pearson_similarity,
    'cosine_similarity': cosine_similarity,
}


def _get_similarity_function(identity):
    return _sim_dict[identity]


def get_similarity_mat(uimat, mode='row', similarity='euclid_similarity'):
    """根据user_item评分矩阵，计算相似度矩阵
    mode: 'row'表示基于用户，'col'表示基于物品
    similarity: 表示采取何种相似度计算方法，支持欧氏距离相似度，皮尔逊系数相似度以及余弦相似度
    """
    print '正在根据user_item矩阵计算相似度矩阵'
    if mode not in ['row', 'col']:
        raise Exception('Only support "row" or "col" mode!')

    rows, cols = uimat.shape
    if mode == 'row':
        t = rows
    else:
        t = cols

    simmat = np.ones((t, t), dtype=np.float32)
    similarity = _get_similarity_function(similarity)
    for i in range(t):
        if (i + 1) % 50 == 0 or (i + 1) == t:
            print '计算进度：{}/{}'.format(i + 1, t)
        for j in range(i + 1, t):
            if mode == 'row':
                simmat[i][j] = simmat[j][i] = similarity(uimat[i, :], uimat[j, :])
            else:
                simmat[i][j] = simmat[j][i] = similarity(uimat[:, i], uimat[:, j])
    return simmat


def estimate_rating_based_on_item(uimat, simmat, user_id, item_id):
    """评估用户user_id对物品item_id的打分est_rating"""
    nb_items = uimat.shape[1]
    est_rating = 0.0
    nb_rating = 0
    for j in range(nb_items):
        rating = uimat[user_id][j]
        if rating == 0:
            continue
        est_rating += simmat[item_id, j] * rating
        nb_rating += 1
    if nb_rating > 0:
        est_rating /= nb_rating

    return est_rating


def recommend(uimat, simmat, user_id, k=5):
    """推荐引擎接口，入口参数分别为事先计算好的user_item矩阵、相似度矩阵、需要推荐的目标用户id以及推荐数目"""
    # print '用户id:', user_id
    unrated_items = np.nonzero(uimat[user_id] == 0)[0]
    # print '未评分物品id:', unrated_items
    if len(unrated_items) == 0:
        print 'You rated everything!'
        return None
    ratings = []
    for item_id in unrated_items:
        rating = estimate_rating_based_on_item(uimat, simmat, user_id, item_id)
        ratings.append((item_id, rating))
    ratings = sorted(ratings, key=lambda kv: kv[1], reverse=True)
    return ratings[:k]


def run():
    # 在简单数据上测试推荐引擎
    uimat = load_simple_data3()
    simmat = get_similarity_mat(uimat, mode='col', similarity='pearson_similarity')
    for i in range(uimat.shape[0]):
        rating = recommend(uimat, simmat, i)
        print 'user id:', i, rating


if __name__ == '__main__':
    run()
