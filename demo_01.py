# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score

from util import LogisticModel, gen_data, add_noise
from noisy_label import loss_unbiased, NoisyLabelModel, make_transition_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--symmetric", action="store_true")
args = parser.parse_args()
np.random.seed(17)

n = 4000
m = 2
if args.symmetric:
    gamma = np.array([0.7, 0.7]) #n_classes must be 2.
    dir_name = os.path.join("result", "symmetric")
else:
    gamma = np.array([0.95, 0.6]) #n_classes must be 2.
    dir_name = os.path.join("result", "asymmetric")

Gamma = make_transition_matrix(gamma)
n_classes = gamma.shape[0]

true_w = np.array([5.0, 0.0])
true_b = 0.0
true_model = LogisticModel(true_w, true_b)

print("n: {}".format(n))
print("true w: {}".format(true_w))
print("true b: {}".format(true_b))
print("true Gamma:\n{}".format(Gamma))

X, y = gen_data(n, m, n_classes, true_model, separable=False, prob_thre=0.1)
y_noisy = add_noise(y, Gamma)
X_test, y_test = gen_data(n, m, n_classes, true_model, separable=False, prob_thre=0.1)
y_noisy_test = add_noise(y_test, Gamma)           

model1 = LogisticModel(np.zeros(2), 0)
model1.fit(X, y_noisy, max_iter=2000)
y_test_pred1 = model1.predict_proba(X_test)

labels = [1] if n_classes == 2 else None
print("cross entropy: {}".format(log_loss(y_test, y_test_pred1)))
print("accuracy: {}".format(accuracy_score(y_test.argmax(1), y_test_pred1.argmax(1))))
print("precision: {}".format(precision_score(y_test.argmax(1), y_test_pred1.argmax(1), average=None, labels=labels)))
print("recall: {}".format(recall_score(y_test.argmax(1), y_test_pred1.argmax(1), average=None, labels=labels)))
print("w: {}".format(model1.w))
print("b: {}".format(model1.b))

model2 = LogisticModel(np.zeros(2), 0)
noisy_label_model = NoisyLabelModel(model2, n_classes=n_classes, noise_model="uniform")
#print(noisy_label_model.gamma0, noisy_label_model.gamma1)
noisy_label_model.fit(X, y_noisy, max_iter=40, arg_fit={"max_iter": 50})
y_test_pred2 = noisy_label_model.infer_clean_label(X_test)
#y_test_pred = noisy_label_model.get_clean_model().predict_proba(X_test)


print("cross entropy: {}".format(log_loss(y_test, y_test_pred2)))
print("accuracy: {}".format(accuracy_score(y_test.argmax(1), y_test_pred2.argmax(1))))
print("precision: {}".format(precision_score(y_test.argmax(1), y_test_pred2.argmax(1), average=None, labels=labels)))
print("recall: {}".format(recall_score(y_test.argmax(1), y_test_pred2.argmax(1), average=None, labels=labels)))
print("w: {}".format(model2.w))
print("b: {}".format(model2.b))
print("Gamma:\n{}".format(noisy_label_model.Gamma))

plt.scatter(x=X[:,0][y_noisy[:,0]==1], y=X[:,1][y_noisy[:,0]==1], color="black", alpha=0.5, label="negative")
plt.scatter(x=X[:,0][y_noisy[:,1]==1], y=X[:,1][y_noisy[:,1]==1], color="red", alpha=0.5, label="positive")

def plot_decision_boundary(model, color, label):
    w = model.w
    b = model.b
    f = lambda y: (- w[1] * y - b) / w[0]
    return plt.plot([f(-1), f(1)], [-1,1], color=color, label=label)

plot_decision_boundary(model1, "blue", "ordinary")
plot_decision_boundary(model2, "yellow", "noise-aware")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc=2)
plt.title("training data with noisy labels and the learned decision boundaries")
#plt.show()
plt.savefig(os.path.join(dir_name, "training_data.png"))
plt.clf()

plt.scatter(x=X_test[:,0][y_test[:,0]==1], y=X_test[:,1][y_test[:,0]==1], color="black", alpha=0.5, label="negative")
plt.scatter(x=X_test[:,0][y_test[:,1]==1], y=X_test[:,1][y_test[:,1]==1], color="red", alpha=0.5, label="positive")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc=2)
plt.title("test data with clean labels")
#plt.show()
#plt.savefig("./result/test_data.png")
plt.savefig(os.path.join(dir_name, "test_data.png"))
plt.clf()

x_plot = np.arange(-2,2,0.05)
X_plot = np.c_[x_plot, np.zeros(x_plot.shape[0])]
plt.plot(x_plot, model1.predict_proba(X_plot)[:,1], color="blue", label="ordinary")
plt.plot(x_plot, model2.predict_proba(X_plot)[:,1], color="yellow", label="noise-aware")
plt.plot(x_plot, true_model.predict_proba(X_plot)[:,1], color="black", label="true")
plt.legend(loc=2)
plt.xlabel("x1")
plt.ylabel("Prob[y=1|x1, x2=0]")
#plt.show()
#plt.savefig("./result/sigmoid_curve.png")
plt.savefig(os.path.join(dir_name, "sigmoid.png"))

