"""
Produces an error report given a sentence index.
Outputs accuracy/most-frequent-prediction per language/tag/word in the sentence.
Also outputs a bracketed syntax tree.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
from utils import *


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_id', type=int)
    return parser

def print_parsed_sentence(first_word_index, sentence_length, conll_file):
    parse = ''
    infile = codecs.open(conll_file, 'r')
    curr_index = 0
    for line in infile:
        # Ignore the same lines that are ignored when originally parsing the CoNLL files.
        if line[0] == '#' or line == '\n' or line.strip() == '':
            continue
        if first_word_index <= curr_index and curr_index < first_word_index + sentence_length: 
            fields = line.split()
            to_add = fields[5].replace('(', '[').replace(')', ']')
            to_add = to_add.replace('*', '[{0} {1}]'.format(fields[4], fields[3]))
            parse = parse + to_add
        curr_index += 1
    print(parse)
    
def print_for_lang_and_tag(y, word_predictions):
    for i in range(len(y)):
        counts = dict()
        for prediction in word_predictions[i]:
            counts.setdefault(prediction, 0)
            counts[prediction] += 1
        most_frequent = max(counts, key=counts.get)
        counts.setdefault(y[i], 0)
        print('Correct ({0}): {1}, Most Frequent ({2}): {3}'.format(y[i], counts[y[i]]/20, most_frequent, counts[most_frequent]/20))

def sentence_error_report(sentence_id, sentences_path, conll_file, predictions_files):
    sentences_file = open(sentences_path, "r")
    all_word_sentences = sentences_file.readlines()
    num_words = len(all_word_sentences)
    
    sentence_indices = get_sentence_indices(sentences_path)
    sentence_indices.append(num_words)
    first_word_index = sentence_indices[sentence_id]
    sentence_length = sentence_indices[sentence_id + 1] - sentence_indices[sentence_id]
    print('SENTENCE: {}'.format(all_word_sentences[first_word_index + sentence_length - 1]))
    
    print_parsed_sentence(first_word_index, sentence_length, conll_file)
    
    tag_types = ['POS', 'Parent', 'Grandparent', 'GGrandparent']
    languages = ['AR', 'ES', 'FR', 'RU', 'ZH']
    for tag_num in range(4):
        print('TAG: {}'.format(tag_types[tag_num]))
        for language_num in range(5):
            print('{}: '.format(languages[language_num]))
            # Indexed by word, experiment.
            word_predictions = []
            for i in range(sentence_length):
                word_predictions.append([])
            for experiment in range(0, 20):
                y, y_hat = read_predictions(predictions_files[language_num][tag_num][experiment])
                y = y[first_word_index:first_word_index + sentence_length]
                y_hat = y_hat[first_word_index:first_word_index + sentence_length]
                for i in range(len(y)):
                    word_predictions[i].append(y_hat[i])                    
            print_for_lang_and_tag(y, word_predictions)


def main(sentence_id):
    # A list of lists of lists containing the predictions files (indices are language, tag, experiment).
    predictions_files = []
    models = ['predictions-ar-decay-model_step_149000', 'predictions-es-decay-model_step_117000', 'predictions-fr-decay-model_step_147000',
             'predictions-ru-decay-model_step_149000', 'predictions-zh-decay-model_step_147000']
    for language_num in range(5):
        predictions_files.append([])
        for tag_type in range(1, 5):
            predictions_files[language_num].append([])
            for experiment in range(1, 21):
                predictions_files[language_num][tag_type-1].append('drive/My Drive/Cog Sci Comps/Syntax Experiment Results/Hidden and Cell Models/experiment{0}/{1}-tag{2}.txt'.format(experiment, models[language_num], tag_type))
    sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    conll_file = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/conll_test.txt'
    sentence_error_report(sentence_id, sentences_path, conll_file, predictions_files)
    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.sentence_id)
