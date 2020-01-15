"""
Combines text files in a directory into one text file.
"""

from __future__ import unicode_literals

import os
import sys
import codecs
import re
import copy
import argparse

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    return parser


def main(directory):
    outfile = codecs.open("output.txt", 'w', encoding='utf-8')
    for root, dirs, files in os.walk(directory):
        line_count = 0
        for fname in files:
            # Can change this to be any substring.
            if not 'gold_conll' in fname:
                continue
            filename = os.path.join(root, fname)
            print("Writing " + filename)
            file = codecs.open(filename, encoding='utf-8')
            for line in file:
                outfile.write(line)
                line_count += 1
        outfile.write('\n')
    print('Done; final line count: {}'.format(line_count))

    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.input_dir)
