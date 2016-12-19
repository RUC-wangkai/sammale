# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: GNU General Public License(Version 3)
@file: optimizers.py
@time: 2016/12/19 18:00


"""


class SGD(object):
    def __init__(self, lr, momentum, decay):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        self.current_lr = lr
        self.current_iteration = 0

    def update(self, W, d_W):
        # TODO
        pass

    def next_iteration(self):
        # TODO
        pass
