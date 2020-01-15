"""
Script to read a CoNLL file.
Generates a sentences text file and a phrase tags text file.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read a CoNLL file.")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input dataset.")

    parser.add_argument(
        '--output_sentences', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output dataset.")
    
    parser.add_argument(
        '--output_phrase_tags', '-w', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output dataset.")
        
    return parser


def main(infile, out_sentences, out_phrase_tags):
    line_count = 0
    curr_sentence = ''
    curr_tags = []
    for line in infile:
        # Ignore some lines.
        if line[0] == '#' or line == '\n' or line.strip() == '':
            continue
        fields = line.split()
        
        # Save word and POS.
        curr_line_out = fields[3] + '\t' + fields[4]
        # Do not include a space before a '.' or ','.
        if fields[4] != '.' and fields[3] != ',':
            curr_sentence = curr_sentence + ' '
        curr_sentence = curr_sentence + fields[3]
        
        parse = fields[5]
        # Read parses.
        curr_tag = ''
        for char in parse:
            # Assume '(' can only appear before '*' and ')' can only appear after.
            if char == '(' or char == '*':
                if curr_tag != '':
                    if curr_tag == 'TOP':
                        # End previous sentence.
                        curr_sentence = fields[3]
                        curr_tags = []
                    # Add a tag and keep reading a new phrase tag.
                    curr_tags.append(curr_tag)
                    curr_tag = ''

            if char == '*':
                # This is the leaf.
                for i in range(3):
                    if len(curr_tags) > i:
                        curr_line_out = curr_line_out + '\t' + curr_tags[-1-i]
                    else:
                        curr_line_out = curr_line_out + '\tNONE'
            elif char == ')':
                curr_tags.pop()
            elif char != '(':
                # Keep reading a phrase tag.
                curr_tag = curr_tag + char
        
        # Clean sentence from contractions / possessives.
        cleaned_sentence = curr_sentence.replace(" '", "'")
        cleaned_sentence = cleaned_sentence.replace(" n't", "n't")
        
        # Write to files.
        if line_count != 0:
            out_sentences.write('\n')
            out_phrase_tags.write('\n')
        out_sentences.write(cleaned_sentence)
        out_phrase_tags.write(curr_line_out)
        line_count += 1
        
    print("Done; wrote {} lines.".format(line_count))


    
if __name__ == '__main__':

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)
    
    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output_sentences.name != '<stdout>':
        args.output_sentences = codecs.open(args.output_sentences.name, 'w', encoding='utf-8')
    if args.output_phrase_tags.name != '<stdout>':
        args.output_phrase_tags = codecs.open(args.output_phrase_tags.name, 'w', encoding='utf-8')

    main(args.input, args.output_sentences, args.output_phrase_tags)
