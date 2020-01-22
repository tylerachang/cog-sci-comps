"""
Run a syntax experiment! Requires the representations tensor (layers x examples x units) and 
the corresponding phrase tag text file. Each row of the phrase tag text file should be
tab-separated values: word POS parent gp ggp.
"""

from nn_model import Experiment
import numpy as np
import torch
import argparse
from utils import *
import pickle
import codecs

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_reps')
    parser.add_argument('--dev_reps')
    parser.add_argument('--test_reps')
    parser.add_argument('--train_tags')
    parser.add_argument('--dev_tags')
    parser.add_argument('--test_tags')
    # Do not use the BPE processed sentences.
    parser.add_argument('--dev_sentences_path')
    parser.add_argument('--test_sentences_path')
    parser.add_argument('--save_model_path')
    parser.add_argument('--output_observations')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--prediction_tag', type=int)
    return parser


def run_experiment(train_reps, dev_reps, test_reps,
         train_tags, dev_tags, test_tags, dev_sentences_path, test_sentences_path,
         save_model_path, output_observations, num_epochs, prediction_tag):
    X_train = load_reps(train_reps)
    X_dev = load_reps(dev_reps)
    X_test = load_reps(test_reps)
    
    n_train = list(X_train.size())[0]
    n_test = list(X_test.size())[0]
    dims = list(X_train.size())[1]
    
    y_train = load_tags(train_tags, prediction_tag)
    y_dev = load_tags(dev_tags, prediction_tag)
    y_test = load_tags(test_tags, prediction_tag)
    
    # All classes that appear in the train or test set.
    classes = list(set(y_train + y_dev + y_test))
    
    # Belinkov (2017) uses 500 hidden dimension and 30 epochs.
    # Blevins (2018) uses 300 hidden dimension.
    experiment = Experiment(classes, input_dim = dims, num_layers = 1, hidden_dims = 500)
    experiment.train(X_train, y_train, max_epochs=num_epochs, X_dev=X_dev, y_dev=y_dev,
					dev_sentences_path=dev_sentences_path, batch_size=64, save_path=save_model_path)

    best_model = pickle.load(open(save_model_path, "rb"))
    y_test_hat = best_model.predict(X_test)
    test_sentence_indices = get_sentence_indices(test_sentences_path)
	
    sentence_accs = sentence_accuracies(y_test_hat, y_test, test_sentence_indices)
    print('Test sentence-averaged accuracy: {}'.format(sum(sentence_accs)/len(sentence_accs)))
    print('Test accuracy: {}'.format(
        accuracy(y_test_hat, y_test)))
    
    outfile = codecs.open(output_observations, 'w', encoding='utf-8')
    for acc in sentence_accs:
        outfile.write(str(acc))
        outfile.write('\n')
    print('Saved observations to {}'.format(output_observations))
    
    
if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run_experiment(args.train_reps, args.dev_reps, args.test_reps,
         args.train_tags, args.dev_tags, args.test_tags,
         args.dev_sentences_path, args.test_sentences_path,
         args.save_model_path, args.output_observations, args.num_epochs, args.prediction_tag)
