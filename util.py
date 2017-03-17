# -*- coding:utf-8 -*-

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


class LogisticModel(object):
    
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict_proba(self, X):
        p1 = expit(X.dot(self.w)+self.b)
        return np.c_[1 - p1, p1]

    def fit(self, X, y, max_iter=100, lr=0.5, l2=0.0):
        n = y.shape[0]
        for i in range(max_iter):
            p = self.predict_proba(X)
            self.w = self.w + lr * ((y[:,1] - p[:,1]).dot(X) / n - l2 * self.w)
            self.b = self.b + lr * np.mean(y[:,1] - p[:,1])

def x_entropy(y1, y2):
    return - np.sum(np.log(y2**y1), 1)

def gen_data(n, m, n_classes, model, separable=True, prob_thre = 0.1):
    X = np.zeros(shape=(n, m))
    ng = np.zeros(shape=n) == 0
    sum_ng = np.sum(ng)
    while(sum_ng > 0):
        X_add = np.random.random(size=(sum_ng, m)) * 2 - 1
        X[ng,:] = X_add
        ng = np.max(model.predict_proba(X), 1) - 1.0 / n_classes < prob_thre
        ng[:n//3] = False
        sum_ng = np.sum(ng)
    if separable:
        idx = model.predict_proba(X).argmax(1)
    else:
        r = np.random.random(n)
        p = model.predict_proba(X).cumsum(1)
        #p[:,-1] = 1
        assert (p >= r[:,np.newaxis]).any(1).all()
        idx = (p >= r[:,np.newaxis]).argmax(1)
    y = np.zeros(shape=(n, n_classes))
    y[np.arange(n), idx] = 1
    return X, y

def add_noise(y, Gamma):
    n = y.shape[0]
    p = (y.dot(Gamma)).cumsum(1)
    #p[:,-1] = 1
    r = np.random.random(n)
    assert (p >= r[:,np.newaxis]).any(1).all()
    idx = (p >= r[:,np.newaxis]).argmax(1)
    y_noisy = np.zeros(y.shape)
    y_noisy[np.arange(n), idx] = 1
    return y_noisy

