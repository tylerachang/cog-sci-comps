"""
Parse the logs outputted by OpenNMT-py.
Reads evaluation accuracies or perplexities.
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
        '--read_perplexity', '-p', type=int, default=0,
        help="Read perplexity or accuracy.")
        
    return parser

def main(infile, outfile, read_perplexity):
    for line in infile:
        if read_perplexity == 0:
            if "Validation accuracy" in line:
                outfile.write(line.split()[-1] + "\n")
        else:
            if "Validation perplexity" in line:
                outfile.write(line.split()[-1] + "\n")

    
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

    main(args.input, args.output, args.read_perplexity)
