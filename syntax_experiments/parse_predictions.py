"""
Reads all the predictions files and takes averages.
Can output an F1 scores file (outputs an F1 score for each individual tag, model, language),
an accuracy per word file (outputs an accuracy for each word, task, language), 
an accuracy per sentence file (outputs an accuracy for each sentence, task, language),
or a confusion matrix.
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

from utils import *


# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file')
    return parser

# Reads a predictions file and outputs y and y_hat lists.
def read_predictions(filepath):
    infile = codecs.open(filepath, 'r')
    y = []
    y_hat = []
    for line in infile:
        if line.strip() == '':
            continue
        prediction = line.split()
        y_hat.append(prediction[0])
        y.append(prediction[1])
    return y, y_hat

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

# Computes and writes the F1 score per class.
def compute_f1_per_class(y, y_hat, outfile, language, tag_type):
    # F1 scores per class.
    f1_dict = {}
    classes = list(set(y))
    classes.sort()
    for tag in classes:
        class_y = []
        class_y_hat = []
        
        # Treat as a binary classification.
        for i in range(len(y)):
            class_y.append(y[i] == tag)
            class_y_hat.append(y_hat[i] == tag)
        class_f1 = metrics.f1_score(class_y, class_y_hat, average='binary', zero_division=0)
        class_precision = metrics.precision_score(class_y, class_y_hat, average='binary', zero_division=0)
        class_recall = metrics.recall_score(class_y, class_y_hat, average='binary', zero_division=0)        
        class_acc = metrics.accuracy_score(class_y, class_y_hat)
        f1_dict[tag] = class_f1
        outfile.write('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(tag_type, language, tag, class_f1, class_precision, class_recall, class_acc))
        
        # Code to compute proportions of each phrase tag.
#        outfile.write('\n{0}\t{1}\t{2}'.format(tag_type, tag, sum(class_y)/len(class_y)))

# Plots the confusion matrix.
def plot_confusion_matrix(y, y_hat):
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

# Gets the language and tag type given the filename.
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
    
    
# Computes and writes accuracies per word.
def compute_accuracies_per_word(predictions_files, sentences_path):
    # Read all the prediction accuracies into a dictionary.
    # Keys of the form language-phraseTag.
    # Values are lists of prediction lists.
    prediction_lists = {}
    for file in predictions_files:
        print('Reading {}'.format(file))
        language, tag_type = get_language_and_tag_type(file)
        prediction_list_key = '{0}-{1}'.format(language, tag_type)
        prediction_list = read_predictions_acc(file)
        prediction_lists.setdefault(prediction_list_key, [])
        prediction_lists[prediction_list_key].append(prediction_list)
    
    for key in prediction_lists.keys():
        print('Writing per word accuracies for {}'.format(key))
        # Create an output file for each language and for each tag.
        outfile = codecs.open('drive/My Drive/Cog Sci Comps/Syntax Experiment Results/Per Word Accuracies/{}-word_accuracies.txt'.format(key), 'w')
        outfile.write('Word ID\tWord\tAccuracy')
        all_predictions = prediction_lists[key]
        
        # Accuracy for each word (each entry in a prediction list).
        sentences = codecs.open(sentences_path, 'r')
        i = 0
        for sentence in sentences:
            accuracy = sum(prediction_list[i] for prediction_list in all_predictions)/len(all_predictions)
            outfile.write('\n{0}\t{1}\t{2}'.format(i, sentence[:-1], accuracy))
            i += 1
        if i != len(prediction_lists[key][0]):
            print('WARNING: Sentences not same length as predictions')
            
            
# Computes and writes accuracies per sentence.
def compute_accuracies_per_sentence(predictions_files, sentences_path):
    sentence_indices = get_sentence_indices(sentences_path)
    sentences_file = open(sentences_path, "r")
    all_sentences = sentences_file.readlines()
    sentences = []
    for index in sentence_indices:
        sentences.append(all_sentences[index-1][:-1])
    # Move the first sentence to the end so that they are in the correct order.
    first = sentences.pop(0)
    sentences.append(first)
    
    # Read all the per-sentence prediction accuracies into a dictionary.
    # Keys of the form language-phraseTag.
    # Values are lists of prediction lists.
    prediction_lists = {}
    for file in predictions_files:
        print('Reading {}'.format(file))
        language, tag_type = get_language_and_tag_type(file)
        prediction_list_key = '{0}-{1}'.format(language, tag_type)
        y, y_hat = read_predictions(file)
        prediction_list = sentence_accuracies(y_hat, y, sentence_indices)
        prediction_lists.setdefault(prediction_list_key, [])
        prediction_lists[prediction_list_key].append(prediction_list)
    
    for key in prediction_lists.keys():
        print('Writing per sentence accuracies for {}'.format(key))
        # Create an output file for each language and for each tag.
        outfile = codecs.open('drive/My Drive/Cog Sci Comps/Syntax Experiment Results/Per Sentence Accuracies/{}-sentence_accuracies.txt'.format(key), 'w')
        outfile.write('Sentence ID\tSentence\tAccuracy')
        all_predictions = prediction_lists[key]
        
        # Accuracy for each sentence (each entry in a prediction list).
        i = 0
        for sentence in sentences:
            accuracy = sum(prediction_list[i] for prediction_list in all_predictions)/len(all_predictions)
            outfile.write('\n{0}\t{1}\t{2}'.format(i, sentence, accuracy))
            i += 1
        if i != len(prediction_lists[key][0]):
            print('WARNING: Sentences not same length as predictions')
        

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
    
    # Compute F1 scores per class.
    f1_outfile = codecs.open('f1_output.txt', 'w')
    f1_outfile.write('Type\tLanguage\tTag\tF1\tPrecision\tRecall\tAcc')
    for file in predictions_files:
        print('Reading {}'.format(file))
        language, tag_type = get_language_and_tag_type(file)
        y, y_hat = read_predictions(file)
        compute_f1_per_class(y, y_hat, f1_outfile, language, tag_type)
#        plot_confusion_matrix(y, y_hat)

    # Compute accuracies per word.
    sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    compute_accuracies_per_word(predictions_files, sentences_path)
    
    # Compute accuracies per sentence.
    sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    compute_accuracies_per_sentence(predictions_files, sentences_path)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.predictions_file)
