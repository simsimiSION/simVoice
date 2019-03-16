#!/usr/bin/env python
# encoding: utf-8
# @author: simsimi
# @contact: dail:911
# @software: pycharm-anaconda3
# @file: svm_classifier.py

from sklearn.svm import SVC
import numpy as np

class svm(object):
    def __init__(self):
        self.svc = SVC()

    def fit(self, x, y):
        y = np.array(y)
        y_shape = y.shape

        if len(y_shape) == 1:
            pass
        elif len(y_shape) == 2:
            y = self.id_format(y)
        else:
            raise Exception('y的格式有误')

        self.svc.fit(x, y)

    def predict(self, x):
        self.svc.predict(x)

    def id_format(self, y):
        return y.argmax(axis=0)


if __name__ == '__main__':
    svm_cls = svm()
    svm_cls.fit(0, [[1, 0], [0,1]])