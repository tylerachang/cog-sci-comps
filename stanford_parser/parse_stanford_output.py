"""
Generates lists of predictions given the Stanford parser output.
Note: should include an empty line at the end of the file (so the last sentence doesn't get cut off).
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--true_tags')
    return parser

def get_predictions(input_path):
    predictions = []
    # One list for each tag type (POS, Parent, Grandparent, GGrandparent).
    for i in range(4):
        predictions.append([])
    infile = codecs.open(input_path, 'r')
    line = ''
    last_line_of_parse = False
    curr_tags = []
    for next_line in infile:
        if next_line.strip() == '':
            # The next line is blank, so it's the last line of the parse.
            last_line_of_parse = True
            # Remove all the close parentheses from the end of the line.
            while line[-1] == ')':
                line = line[:-1]
        else:
            last_line_of_parse = False
        if line.strip() == '':
            line = next_line.strip()
            continue
        
        # Read parses.
        curr_tag = ''
        for char in line:
            # Assume space only appears after a tag.
            if char == ' ' and curr_tag != '':
                # Add a tag and keep reading a new phrase tag.
                # Convert ROOT to TOP.
                if curr_tag == 'ROOT':
                    curr_tag = 'TOP'
                curr_tags.append(curr_tag)
                curr_tag = ''
            if char == ')':
                # Pop the most recent tag. Might be reading a word in curr_tag (if we are within a leaf),
                # but then we should clear curr_tag and ignore it anyways.
                curr_tags.pop()                
                curr_tag = ''
            if char != ')' and char != '(' and char != ' ':
                # Keep reading a phrase tag.
                curr_tag = curr_tag + char
                
        if last_line_of_parse:
            # Add the tags to the predictions, then clear.
            for i in range(4):
                if len(curr_tags) > i:
                    predictions[i].append(curr_tags[-1-i])
                else:
                    predictions[i].append('NONE')
            curr_tags = []
            print('Parsed {}'.format(len(predictions[0])))
        elif curr_tag != '':
            # Add the current tag if we are going to continue the current parse.
            if curr_tag == 'ROOT':
                curr_tag = 'TOP'
            curr_tags.append(curr_tag)
        # Set the next line to be the current line.
        line = next_line.strip()
    return predictions


def main(input_path, true_tags_path):
    predictions = get_predictions(input_path)
    true_tags_file = open(true_tags_path, "r")
    true_tags_lines = true_tags_file.readlines() 
    # Output predictions to files.
    for i in range(4):
        outfile = codecs.open('predictions-PCFGSentenceTrainedParser-tag{}.txt'.format(i+1), 'w')
        for word_index in range(len(predictions[i])):
            outfile.write('{0}\t{1}\n'.format(predictions[i][word_index], true_tags_lines[word_index].split()[i+1]))
    

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input, args.true_tags)
