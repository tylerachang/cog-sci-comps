"""
Display a confusion matrix given a predictions file.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import sklearn.metrics as metrics
import numpy as np


# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file')
    return parser

def main(predictions_file):
    infile = codecs.open(predictions_file, 'r')
    
    y = []
    y_hat = []
    for line in infile:
        if line.strip() == '':
            continue
        prediction = line.split()
        y_hat.append(prediction[0])
        y.append(prediction[1])
    
    # F1 scores per class.
    f1_dict = {}
    classes = list(set(y))
    classes.sort()
    for tag in classes:
        class_y = []
        class_y_hat = []
        for i in range(len(y)):
            if y[i] == tag:
                class_y.append(y[i])
                class_y_hat.append(y_hat[i])
        # Use micro average because classes are imbalanced.
        class_f1 = metrics.f1_score(class_y, class_y_hat, average='micro')
        f1_dict[tag] = class_f1
#        print('{0} F1: {1}'.format(tag, class_f1))
    print(sorted(f1_dict, key=f1_dict.get))
    
    # Plot matrix.
    confusion = confusion_matrix(y, y_hat)
    # Normalize each row (actual, along axis 1) or column (predicted, along axis 0).
#    confusion = normalize(confusion, axis=1, norm='l1')
    heatmap = plt.imshow(confusion, cmap = plt.cm.Purples)
    plt.colorbar(heatmap)
    labels = np.unique(y + y_hat)
    axis_values = list(range(labels.size))
    plt.xlabel('Predicted')
    plt.xticks(axis_values, labels, rotation='vertical')
    plt.ylabel('Actual')
    plt.yticks(axis_values, labels)
    plt.show()
#    print(confusion)

    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.predictions_file)
