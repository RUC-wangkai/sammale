关于本项目：
1. 本项目是一个PyCharm项目，可使用PyCharm IDE打开。
2. 本项目下的文件描述：
    CFR.py: Collaborative Filtering Recommender，包含了基于协同过滤的推荐算法的核心代码。
    run_CFR_on_MovieLens.py: 在MovieLens-100k数据集上跑CFR算法的脚本。
    run_LFM_on_MovieLens.py: 在MovieLens-100k数据集上跑LFM算法的脚本。并含有训练LFM模型的代码在内。
    plot_result.py: 将实验结果绘图的脚本。
    u.data: MovieLens-100k数据文件之一。更多其他关于MovieLens-100k数据集的信息可在
        http://grouplens.org/datasets/movielens 上查看。
    train.data与test.data: 训练文件与测试文件。可使用regenerate_movielen_dataset()方法进行重新划分。
