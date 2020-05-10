"""
Run baselines for the syntax experiments. Each row of the phrase tag text file should be
tab-separated values: word POS parent gp ggp.
"""

import argparse
from utils import *

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tags')
    parser.add_argument('--test_tags')
    parser.add_argument('--test_sentences_path')
    parser.add_argument('--prediction_tag', type=int)
    return parser

def load_words_and_tags(tags_path, prediction_tag):
    tag_file = open(tags_path, "r")
    tag_lines = tag_file.readlines()
    y = []
    for line in tag_lines:
        tags = line.split()
        y.append((tags[0], tags[prediction_tag]))
    return y

def most_frequent(items):
    # Returns the most frequent item in the list.
    counts = dict()
    for item in items:
        counts.setdefault(item, 0)
        counts[item] += 1
    return max(counts,key=counts.get)

def most_frequent_tag(xy_train, x_test):
    y_train = [tag for (word, tag) in xy_train]
    mft = most_frequent(y_train)
    predictions = []
    for tag in x_test:
        predictions.append(mft)
    return predictions
    
def per_word_mft(xy_train, x_test):
    # Map each word to a list of tags.
    word_tags_dict = dict()
    for (word, tag) in xy_train:
        word_tags_dict.setdefault(word, [])
        word_tags_dict[word].append(tag)

    # Map each word to its most frequent tag.
    word_tag_dict = dict()
    for word in word_tags_dict.keys():
        word_tag_dict[word] = most_frequent(word_tags_dict[word])

    # Get the overall MFT to use for novel words.
    y_train_tags = [tag for (word, tag) in xy_train]
    mft = most_frequent(y_train_tags)

    # Make predictions.
    predictions = []
    for word in x_test:
        guess = ''
        if word in word_tag_dict.keys():
            guess = word_tag_dict[word]
        else:
            guess = mft
        predictions.append(guess)
 
    return predictions
    

# Prediction tag is an integer 1-4.
def main(train_tags, test_tags, test_sentences_path, prediction_tag):
    
    xy_train = load_words_and_tags(train_tags, prediction_tag)
    x_test = load_tags(test_tags, 0)
    y_test = load_tags(test_tags, prediction_tag)
    
    mft_y_hat = most_frequent_tag(xy_train, x_test)
    per_word_mft_y_hat = per_word_mft(xy_train, x_test)
    
    # Output predictions for per-word MFT.
    predictions = per_word_mft_y_hat
    true_tags_file = open(test_tags, "r")
    true_tags_lines = true_tags_file.readlines()
    outfile = codecs.open('predictions-baseline-tag{}.txt'.format(prediction_tag), 'w')
    for word_index in range(len(predictions)):
        outfile.write('{0}\t{1}\n'.format(predictions[word_index], true_tags_lines[word_index].split()[prediction_tag]))
    
    # Can replace with regular accuracy instead.
    print('MFT sentence-averaged accuracy: {}'.format(
        sentence_averaged_accuracy(mft_y_hat, y_test, get_sentence_indices(test_sentences_path))))
    print('Per-word MFT sentence-averaged accuracy: {}'.format(
        sentence_averaged_accuracy(per_word_mft_y_hat, y_test, get_sentence_indices(test_sentences_path))))
    
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.train_tags, args.test_tags, args.test_sentences_path, args.prediction_tag)
