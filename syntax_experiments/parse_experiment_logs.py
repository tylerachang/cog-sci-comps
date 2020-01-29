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
        description="Parse an experiment log.")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text file.")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output text file.")
        
    return parser

def main(infile, outfile):
    # Store as matrix: tag x languages.
    accs = []
    for i in range(4):
        accs.append([0,0,0,0,0,0])
    tag = 0
    lang = 0
    for line in infile:
        if 'Test accuracy' in line:
            accs[tag][lang] = line.split()[-1]
            
            # Assume loops over tags for each lang.
            if tag == 3:
                tag = 0
                lang += 1
            else:
                tag += 1
                
    # Write the output.
    for accs_for_tag in accs:
        for acc in accs_for_tag:
            outfile.write(acc)
            outfile.write('\t')
        outfile.write('\n')

    
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

    main(args.input, args.output)
