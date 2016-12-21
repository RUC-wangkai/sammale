# encoding: utf-8

import numpy as np


class BatchIterator(object):

    def __init__(self, x, y, batch_size=1, shuffle=True):
        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._is_shuffled = shuffle

        n = x.shape[0]
        self._random_index = np.arange(n)

        self._number_of_batches = int(np.ceil(n * 1. / batch_size))
        self._current_number_of_batches = 0
        self.reallocate()

    def next_batch(self):
        if not self.has_next_batch():
            raise Exception('There is not next batch!')

        idx = self._current_number_of_batches * self._batch_size

        batch_x = self._x[self._random_index[idx:idx + self._batch_size]]
        batch_y = self._y[self._random_index[idx:idx + self._batch_size]]
        self._current_number_of_batches += 1
        return batch_x, batch_y

    def has_next_batch(self):
        return self._current_number_of_batches != self._number_of_batches

    def number_of_batches(self):
        return self._number_of_batches

    def reallocate(self):
        if self._is_shuffled:
            np.random.shuffle(self._random_index)
        self._current_number_of_batches = 0
