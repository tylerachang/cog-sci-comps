"""
Script to read a CoNLL file.
Generates a file containing tree parses.
"""

import sys
import codecs
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read a CoNLL file.")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input dataset.")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output dataset.")
        
    return parser


def main(infile, outfile):
    line_count = 0
    curr_parse = ''
    for line in infile:
        # Ignore some lines.
        if line[0] == '#' or line == '\n' or line.strip() == '':
            continue
        fields = line.split()
        
        if 'TOP' in fields[5] and line_count != 0:
            # Write the current parse for each sentence.
#            outfile.write('{}\n'.format(curr_parse))
            curr_parse = ''
            
        parse_to_add = fields[5].replace('*', '({0} {1})'.format(fields[4], fields[3]))
        curr_parse = curr_parse + parse_to_add
        
        num_close_paren_to_add = curr_parse.count('(') - curr_parse.count(')')
        to_write = curr_parse
        for i in range(num_close_paren_to_add):
            to_write = to_write + ')'
        outfile.write('{}\n'.format(to_write))
        
        line_count += 1
    
    # Only need this if writing parses for each sentence.
#    outfile.write('{}\n'.format(curr_parse))   
    print("Done; read {} lines.".format(line_count))


    
if __name__ == '__main__':
    
    parser = create_parser()
    args = parser.parse_args()

    args.input = codecs.open(args.input.name, encoding='utf-8')
    args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    main(args.input, args.output)
