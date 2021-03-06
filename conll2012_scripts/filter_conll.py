"""
Script to read a CoNLL file.
Extracts every k sentences of a CoNLL file.
Built off of read_conll.py.
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
        description="Filter a CoNLL file.")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input dataset.")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output dataset.")
        
    parser.add_argument(
        '--keep_every', '-k', type=int, default=1,
        help="Keep one in every k sentences.")
        
    return parser


def main(infile, output, keep_every):
    total_line_count = 0
    output_line_count = 0
    curr_tags = []
    curr_sentence_num = 0
    for line in infile:
        # Ignore some lines.
        if line[0] == '#' or line == '\n' or line.strip() == '':
            continue
        fields = line.split()
        
        parse = fields[5]
        # Read parses.
        curr_tag = ''
        for char in parse:
            # Assume '(' can only appear before '*' and ')' can only appear after.
            if char == '(' or char == '*':
                if curr_tag != '':
                    if curr_tag == 'TOP':
                        # End previous sentence.
                        curr_tags = []
                        curr_sentence_num += 1
                    # Add a tag and keep reading a new phrase tag.
                    curr_tags.append(curr_tag)
                    curr_tag = ''

            if char == ')':
                curr_tags.pop()
            elif char != '(':
                # Keep reading a phrase tag.
                curr_tag = curr_tag + char
        
        # Write to files.
        if curr_sentence_num % keep_every == 0:
            if output_line_count != 0:
                output.write('\n')
            output.write(line.strip())
            output_line_count += 1
        total_line_count += 1
        
    print("Done; wrote {0} out of {1} lines.".format(output_line_count, total_line_count))


    
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
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    main(args.input, args.output, args.keep_every)
