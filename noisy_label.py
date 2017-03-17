# -*- coding:utf-8 -*-

from __future__ import print_function

import numpy as np
from sklearn.metrics import log_loss

def make_transition_matrix(gamma, n_classes=None):
    d = np.array(gamma).ndim
    if d == 0:
        Gamma = (1 - gamma) / (1 - n_classes) * np.ones((n_classes, n_classes))
        np.fill_diagonal(Gamma, gamma)
    elif d == 1:
        n_classes = gamma.shape[0]
        Gamma = ((1 - gamma) / (n_classes - 1.0))[:,np.newaxis] * np.ones(n_classes)
        np.fill_diagonal(Gamma, gamma)
    else: 
        Gamma = gamma
    return Gamma

class NoisyLabelModel(object):

    def __init__(self, clean_model, n_classes, noise_model="symmetric", gamma=1.0):
        """
        noise_model: symmetric  The flipping probability Prob[y_clean -> y_noisy] is independent of y_clean and y_noisy.
                     uniform    The flipping probability Prob[y_clean -> y_noisy] is depend on y_clean but independen of y_noisy.
                     nonuniform The flipping probability Prob[y_clean -> y_noisy] is depend on y_clean and y_noisy.
        """
        self.clean_model = clean_model
        self.n_classes = n_classes
        self.noise_model = noise_model
        self.Gamma = make_transition_matrix(gamma, n_classes)

    def get_clean_model(self):
        return self.clean_model
        
    def predict_proba(self, X):
        """ predict probability that a noisy label is 1"""
        y_pred = self.clean_model.predict_proba(X)
        return y_pred.dot(self.Gamma)

    def infer_clean_label(self, X, y=None):
        """ infer true labels from X and noisy labels y """
        y_pred = self.clean_model.predict_proba(X)
        if y is None:
            return y_pred
        prob_to_y = y.dot(self.Gamma.transpose())
        joint_prob = y_pred * prob_to_y
        return y_pred, joint_prob / joint_prob.sum(1)[:,np.newaxis]

    def log_likelihood(self, X, y):
        """ return average log likelihood per data point.
        y: noisy labels
        """ 
        y_pred = self.predict_proba(X)
        return log_loss(y, y_pred)

    def estimate_flip_proba(self, true_labels, noisy_labels):
        contingency = true_labels.transpose().dot(noisy_labels)
        if self.noise_model == "symmetric":
            g = np.diag(contingency).sum() / n
        elif self.noise_model == "uniform":
            g = np.diag(contingency) / contingency.sum(1)
        else:
            g = g
        Gamma = make_transition_matrix(g, self.n_classes)
        return Gamma
        
    def fit(self, X, y, estimates_flip_proba=True, max_iter=50, init_model=True, init_flip_proba=True, arg_fit=None, verbose=False):
        """ fit model to data with an EM algorithm.
        The algorithm is proposed in [1], which considers binary classification and 
        uses multiple noisy labels for each data point.
        (Initialization is not the same as that proposed in [1]. See below.)
        y: noisy labels.
        init_model: initialize parameters of self.clean_model by fitting it to noisy labels if init_model is True.  
        init_flip_proba: initialize flipping probabilities by using error probabilities of the initial clean model if init_flip_proba is True.  
        arg_fit: dictionary of arguments passed to self.clean_model.fit besides X and y.
    
        [1] Rayker et al., "Learning From Crowds", 2010,
            http://www.jmlr.org/papers/volume11/raykar10a/raykar10a.pdf
        """
        n = y.shape[0]
        n1 = np.sum(y)
        n0 = n - n1
        if arg_fit is None:
            arg_fit = {}

        #initialization
        if init_model:
            self.clean_model.fit(X, y, **arg_fit)
        if init_flip_proba:
            r = self.clean_model.predict_proba(X)
            self.Gamma = self.estimate_flip_proba(r, y)

        for i in range(max_iter):
            #e-step
            y_pred, y_pred_post = self.infer_clean_label(X, y)
            if verbose:
                y_pred_noisy = y_pred.dot(self.Gamma)
                print("{} iterations: cross entropy {}".format(i, log_loss(y, y_pred_noisy)))

            #m-step
            self.clean_model.fit(X, y_pred_post, **arg_fit)
            if estimates_flip_proba:
                self.Gamma = self.estimate_flip_proba(y_pred_post, y)
                    
def loss_unbiased(loss, y, y_pred, Gamma):
    """ return an unbiased estimate of loss.
    An unbiased estimator for binary classification is proposed in [1]. This is its generalization to multiclass problems.
    loss: loss function (noisy_label, prediction) -> loss
    y: noisy labels
    y_pred: predictions:
    Gamma: transition matrix (Gamma[i,j] = Prob[y_noisy=j | y_clean=i])
    [1] Natarajan et al., "Learning with noisy labels", 2013, 
        https://papers.nips.cc/paper/5073-learning-with-noisy-labels.pdf

    TODO: Test.
    """
    L = np.zeros(y.shape)
    for i in range(y.shape[1]):
        yi = np.zeros(y.shape)
        yi[:,i] = 1
        L[:,i] = loss(yi, y_pred)
    return (y.transpose().dot(L) * np.linalg.inv(Gamma)).sum()
