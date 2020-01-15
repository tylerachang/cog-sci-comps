"""
Run a syntax experiment! Requires the representations tensor (layers x examples x units) and 
the corresponding phrase tag text file. Each row of the phrase tag text file should be
tab-separated values: word POS parent gp ggp.
"""

from nn_model import Experiment
import numpy as np
import torch

X_train = torch.load('reps_tensor.pt')
X_test = torch.load('reps_tensor_test.pt')

# Concatenate the last layer for hidden and context.
X_train = torch.cat((X_train[3, :, :], X_train[7, :, :]), dim=1)
X_test = torch.cat((X_test[3, :, :], X_test[7, :, :]), dim=1)

n_train = list(X_train.size())[0]
n_test = list(X_test.size())[0]
d = list(X_test.size())[1]

# Read tag text file.
tag_file = open("phrase_tags.txt", "r")
tag_lines = tag_file.readlines()
y_train = []
for line in tag_lines:
	tags = line.split()
	y_train.append(tags[4])
tag_file_test = open("phrase_tags_test.txt", "r")
tag_lines_test = tag_file_test.readlines()
y_test = []
for line in tag_lines_test:
	tags = line.split()
	y_test.append(tags[4])

# All classes that appear in the train or test set.
classes = list(set(y_train + y_test))

# Belinkov (2017) uses 500 hidden dimension and 30 epochs.
# Blevins (2018) uses 300 hidden dimension.
exp = Experiment(classes, input_dim = d, num_layers = 1, hidden_dims = 500)
print('Experiment ', exp)

exp.train(X_train, y_train, max_epochs=25)

# Metrics should all be equal to accuracy.
print('Test set metrics ', exp.metrics(X_test, y_test))
print('Train set metrics ', exp.metrics(X_train, y_train))

tag_counts = dict()
for tag in y_train:
    tag_counts.setdefault(tag, 0)
    tag_counts[tag] += 1
mft = max(tag_counts,key=tag_counts.get)

test_score = 0
for tag in y_test:
    if tag == mft:
        test_score += 1
print('MFT accuracy: {}'.format(test_score/len(y_test)))