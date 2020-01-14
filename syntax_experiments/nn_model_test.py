"""
Majority of this code taken from Deep RNNs Encode Soft Hierarchical Syntax (Blevins, Levy, & Zettlemoyer, 2018). Tests the feedforward NN model used in syntax experiments.
"""

from nn_model import Experiment
import numpy as np
import torch


#test emedding experiments on synthetic data
classes = [-1, 1]

n_train = 7000
n_test = 300 #length for both dev and test sets
d = 50

X_train = np.matrix(np.random.randn(n_train, d))
X_dev = np.matrix(np.random.randn(n_test, d))
X_test = np.matrix(np.random.randn(n_test, d))

w = np.transpose(np.matrix([np.random.rand() for i in range(d)]))

e_train = np.transpose(np.matrix(np.random.randn(n_train)))
e_dev = np.transpose(np.matrix(np.random.randn(n_test)))
e_test = np.transpose(np.matrix(np.random.randn(n_test)))

y_train = np.sign(X_train*w+e_train)
y_dev = np.sign(X_dev*w+e_dev)
y_test = np.sign(X_test*w+e_test)

y_train = [1 if i == 0 else ((float) (i)) for i in y_train]
y_dev = [1 if i == 0 else ((float) (i)) for i in y_dev]
y_test = [1 if i == 0 else ((float) (i)) for i in y_test]

X_train = torch.from_numpy(X_train).float()
X_dev = torch.from_numpy(X_dev).float()
X_test = torch.from_numpy(X_test).float()

exp = Experiment(classes, input_dim = d, num_layers = 1, hidden_dims = d)
print('Experiment ', exp)

exp.train(X_train, y_train, max_epochs=50)

# Metrics should all be equal to accuracy because using binary variables.
print('Test set metrics ', exp.metrics(X_test, y_test))