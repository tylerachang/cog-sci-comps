"""
Run baselines for the syntax experiments. Each row of the phrase tag text file should be
tab-separated values: word POS parent gp ggp.
"""

import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tags')
    parser.add_argument('--dev_tags')
    parser.add_argument('--test_tags')
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

def most_frequent_tag(y_train, y_test):
    y_train = [tag for (word, tag) in y_train]
    y_test = [tag for (word, tag) in y_test]
    mft = most_frequent(y_train)
    test_score = 0
    for tag in y_test:
        if tag == mft:
            test_score += 1
    print('MFT accuracy: {}'.format(test_score/len(y_test)))
    
def per_word_mft(y_train, y_test):
    # Map each word to a list of tags.
    word_tags_dict = dict()
    for (word, tag) in y_train:
        word_tags_dict.setdefault(word, [])
        word_tags_dict[word].append(tag)
    
    # Map each word to its most frequent tag.
    word_tag_dict = dict()
    for word in word_tags_dict.keys():
        word_tag_dict[word] = most_frequent(word_tags_dict[word])

    # Get the overall MFT to use for novel words.
    y_train_tags = [tag for (word, tag) in y_train]
    mft = most_frequent(y_train_tags)

    # Make predictions.
    test_score = 0
    for (word, tag) in y_test:
        guess = ''
        if word in word_tag_dict.keys():
            guess = word_tag_dict[word]
        else:
            guess = mft
        if tag == guess:
            test_score += 1
    print('Per-word MFT accuracy: {}'.format(test_score/len(y_test)))

    
def main(train_tags, dev_tags, test_tags, prediction_tag):
    
    y_train = load_words_and_tags(train_tags, prediction_tag)
    y_dev = load_words_and_tags(dev_tags, prediction_tag)
    y_test = load_words_and_tags(test_tags, prediction_tag)
    
    # All classes that appear in the train or test set.
    classes = list(set(y_train + y_dev + y_test))
    
    most_frequent_tag(y_train, y_test)
    per_word_mft(y_train, y_test)
    
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args.train_tags, args.dev_tags, args.test_tags, args.prediction_tag)
