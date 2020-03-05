"""
Combines accuracy files output by parse_predictions.py.
Also adds the source by reading the original combined CoNLL-2012 file.
Each accuracy file (tab-delimited) should only contain results from one language.
Each accuracy file should have a header, and columns should be ID, sentence, accuracy.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse

from utils import *
        

def main(reading_sentence_accuracies):    
    # Read the sources.
    sources = []
    infile = codecs.open('drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/conll_test.txt', 'r')
    for line in infile:
        # Ignore the same lines that are ignored when originally parsing the CoNLL files.
        if line[0] == '#' or line == '\n' or line.strip() == '':
            continue
        fields = line.split()
        sources.append(fields[0])
        
    # Read the sentences.
    sentences_path = 'drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt'
    sentences_file = open(sentences_path, "r")
    sentences = sentences_file.readlines()
    sentence_indices = get_sentence_indices(sentences_path)
    if not reading_sentence_accuracies:
        sentence_indices = range(len(sentences))
    
    # Clean the sentences and only use the ones corresponding to desired indices.
    final_sources = []
    final_sentences = []
    for index in sentence_indices:
        final_sentences.append(sentences[index-1].strip())
        final_sources.append(sources[index])
    # Move the first sentence to the end so that they are in the correct order.
    first = final_sentences.pop(0)
    final_sentences.append(first)
    
    sources = final_sources
    sentences = final_sentences
      
    # Read accuracy data (a list of lists of lists).
    # Indexed by [language][tag][word/sentence]
    accuracy_data = []
    tags = ['POS', 'Parent', 'Grandparent', 'GGrandparent']
    languages = ['AR', 'EN', 'ES', 'FR', 'RU', 'ZH']
    language_num = 0
    for language in languages:
        accuracy_data.append([])
        tag_num = 0
        for tag in tags:
            accuracy_data[language_num].append([])
            accuracies_file = ''
            if reading_sentence_accuracies:
                accuracies_file = 'drive/My Drive/Cog Sci Comps/RESULTS/Per Sentence Accuracies/{0}-{1}-sentence_accuracies.txt'.format(language, tag)
            else:
                accuracies_file = 'drive/My Drive/Cog Sci Comps/RESULTS/Per Word Accuracies/{0}-{1}-word_accuracies.txt'.format(language, tag)
            
            # Read the accuracies.
            accuracy_list = []
            infile = codecs.open(accuracies_file, 'r')
            line_count = 0
            for line in infile:
                if line_count == 0:
                    line_count += 1
                    continue
                fields = line.split('\t')
                if len(fields) != 3:
                    print('WARNING: more than three fields in accuracy file.')
                accuracy_list.append(float(fields[2]))
            
            if len(accuracy_list) != len(final_sentences):
                print('WARNING: accuracies and sentences/sources have different lengths.')
            accuracy_data[language_num][tag_num] = accuracy_list
            
            tag_num += 1
        language_num += 1
        
    # Write the data to the output files (one file for each tag type).
    tag_num = 0
    for tag in tags:
        outfile_path = 'drive/My Drive/Cog Sci Comps/RESULTS/{}_sentence_accuracies.txt'.format(tag) if reading_sentence_accuracies else 'drive/My Drive/Cog Sci Comps/RESULTS/{}_word_accuracies.txt'.format(tag) 
        outfile = codecs.open(outfile_path, 'w')
        outfile.write('ID\tSource\tString\tAR\tEN\tES\tFR\tRU\tZH')
        
        for i in range(len(sources)):
            outfile.write('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}'.format(i, sources[i], sentences[i],
                    accuracy_data[0][tag_num][i], accuracy_data[1][tag_num][i], accuracy_data[2][tag_num][i],
                    accuracy_data[3][tag_num][i], accuracy_data[4][tag_num][i], accuracy_data[5][tag_num][i]))
        outfile.close()
        tag_num += 1


if __name__ == '__main__':
    reading_sentence_accuracies = False
    main(reading_sentence_accuracies)
