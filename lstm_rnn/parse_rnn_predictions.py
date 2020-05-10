"""
Reads all the predictions files and takes averages.
Can output an accuracy per word file (outputs an accuracy for each word, task), 
an accuracy per sentence file (outputs an accuracy for each sentence, task),
and an accuracy per model file (outputs an accuracy for each model, task).
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
import sklearn.metrics as metrics
import numpy as np

# Reads a predictions file and outputs a list of 0s and 1s (incorrect and correct).
def read_predictions_acc(filepath):
    infile = codecs.open(filepath, 'r')
    acc = []
    for line in infile:
        if line.strip() == '':
            continue
        prediction = line.split()
        acc.append(prediction[0] == prediction[1])
    return acc

# Reads an observations file and outputs a list of the accuracies (real numbers 0-1).
def read_accs(filepath):
    infile = codecs.open(filepath, 'r')
    acc = []
    for line in infile:
        if line.strip() == '':
            continue
        acc.append(float(line.strip()))
    return acc
    
# Computes and writes accuracies per row (row_type = 'observations' or 'predictions').
def compute_accuracies_per_row(row_type):
    # Read all the prediction accuracies into a dictionary.
    # Keys of the form phraseTag (number 1 to 4).
    # Values are lists of prediction lists.
    prediction_lists = {}
    for tag_num in range(1,5):
        for experiment in range(1,11):
            file = 'drive/My Drive/Cog Sci Comps/RNN Experiments/experiment{0}/{1}-rnn-model-tag{2}.txt'.format(experiment, row_type, tag_num)
            print('Reading {}'.format(file))
            if row_type == 'observations':
                prediction_list = read_accs(file)
            else:
                prediction_list = read_predictions_acc(file)
            prediction_lists.setdefault(tag_num, [])
            prediction_lists[tag_num].append(prediction_list)
    
    tag_types = ['POS', 'Parent', 'Grandparent', 'GGrandparent']
    for key in prediction_lists.keys():
        print('Writing per row accuracies for {}'.format(key))
        if row_type == 'observations':
            row_name = 'sentence'
        else:
            row_name = 'word'
        # Create an output file for each language and for each tag.
        outfile = codecs.open('drive/My Drive/Cog Sci Comps/RESULTS/RNN Results/RNN-{0}_{1}_accuracies.txt'.format(tag_types[key-1], row_name), 'w')
        outfile.write('ID\tAccuracy')
        all_predictions = prediction_lists[key]
        
        for i in range(len(all_predictions[0])):
            accuracy = sum(prediction_list[i] for prediction_list in all_predictions)/len(all_predictions)
            outfile.write('\n{0}\t{1}'.format(i, accuracy))

def compute_accuracies_per_model(outfile_path):
    tag_types = ['POS', 'Parent', 'Grandparent', 'GGrandparent']
    outfile = codecs.open(outfile_path, 'w')
    outfile.write('Tag\tModel\tAcc')
    
    for tag_num in range(1,5):
        for experiment in range(1,11):
            file = 'drive/My Drive/Cog Sci Comps/RNN Experiments/experiment{0}/predictions-rnn-model-tag{1}.txt'.format(experiment, tag_num)
            print('Reading {}'.format(file))
            prediction_list = read_predictions_acc(file)
            accuracy = sum(prediction_list)/len(prediction_list)
            outfile.write('\n{0}\tRNN\t{1}'.format(tag_types[tag_num-1], accuracy))
    
            
def main():
    # Compute accuracies per word.
    compute_accuracies_per_row('predictions')
    
    # Compute accuracies per sentence.
    compute_accuracies_per_row('observations')

    # Compute accuracies per model.
    compute_accuracies_per_model('drive/My Drive/Cog Sci Comps/RESULTS/RNN Results/rnn_accuracies.txt')

if __name__ == '__main__':
    main()
