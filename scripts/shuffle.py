"""
1. Create a permutation of the indices.
2. Shuffle a text file according to the permutation.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
import random
import pickle

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse an OpenNMT-py log.")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text file.")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output text file.")
    
    parser.add_argument(
        '--create_permutation', '-p', type=int, default=0,
        help="Create a permutation or shuffle files.")
        
    return parser

def create_permutation(infile):
    print("Reading file...\n")
    i = 0
    permutation = []
    for line in infile:
        permutation.append(i)
        i += 1
    print("Shuffling...\n")
    random.shuffle(permutation)
    pickle.dump(permutation, open("permutation.pickle", "wb"))
    print("Done")

def shuffle_file(infile, outfile):
    permutation = pickle.load(open("permutation.pickle", "rb" ))
    print("Reading file...\n")
    lines = []
    for line in infile:
        lines.append(line)
    print("Writing output file...\n")
    for index in permutation:
        outfile.write(lines[index])
    print("Done")
    
def main(infile, outfile, int_create_permutation):
    if int_create_permutation != 0:
        create_permutation(infile)
    else:
        shuffle_file(infile, outfile)
    
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

    main(args.input, args.output, args.create_permutation)
