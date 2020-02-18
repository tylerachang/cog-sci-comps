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

def compute_f1(infile, outfile, language, tag_type):
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
        outfile.write('\n{0}\t{1}\t{2}\t{3}'.format(tag_type, language, tag, class_f1))
#        print('{0} F1: {1}'.format(tag, class_f1))
#    print(sorted(f1_dict, key=f1_dict.get))
    
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

def get_language_and_tag_type(filename):
    language = 'X'
    if 'predictions-es' in filename:
        language = 'ES'
    elif 'predictions-en' in filename:
        language = 'EN'
    elif 'predictions-ar' in filename:
        language = 'AR'
    elif 'predictions-ru' in filename:
        language = 'RU'
    elif 'predictions-fr' in filename:
        language = 'FR'
    elif 'predictions-zh' in filename:
        language = 'ZH'
    tag_type = 'X'
    if 'tag1' in filename:
        tag_type = 'POS'
    elif 'tag2' in filename:
        tag_type = 'Parent'
    elif 'tag3' in filename:
        tag_type = 'Grandparent'
    elif 'tag4' in filename:
        tag_type = 'GGrandparent'
    return language, tag_type
        

def main(predictions_file):
    predictions_files = []
    if predictions_file == None or predictions_file == '':
        models = ['predictions-es-decay-model_step_117000', 'predictions-fr-decay-model_step_147000',
                 'predictions-en-decay-model_step_150000', 'predictions-ar-decay-model_step_149000',
                 'predictions-ru-decay-model_step_149000', 'predictions-zh-decay-model_step_147000']
        for experiment in range(1, 21):
            for model in models:
                for tag_type in range(1, 5):
                    predictions_files.append('drive/My Drive/Cog Sci Comps/Syntax Experiment Results/Hidden and Cell Models/experiment{0}/{1}-tag{2}.txt'.format(experiment, model, tag_type))
    else:
        predictions_files.append(predictions_file)
        
    outfile = codecs.open('output.txt', 'w')
    outfile.write('Type\tLanguage\tTag\tF1')
    for file in predictions_files:
        language, tag_type = get_language_and_tag_type(file)            
        infile = codecs.open(file, 'r')
        compute_f1(infile, outfile, language, tag_type)

    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.predictions_file)
