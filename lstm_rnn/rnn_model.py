"""
LSTM RNN that takes sentence inputs (tokens separated by spaces) and outputs a classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

# Copy utils from syntax_experiments directory.
from utils import *


# GLOBAL VARIABLES.
EMBEDDING_DIM = 500
HIDDEN_DIM = 500
NUM_HIDDEN = 4
LEARNING_RATE = 0.0002
MAX_EPOCHS = 1000
BATCH_SIZE = 64


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def create_input_tensor_batches(training_data, word_to_ix, batch_size, include_last_batch):
    sentences = []
    batch_max_lengths = []
    curr_max_length = 0
    for i in range(len(training_data)):
        sentence = training_data[i][0]
        sentences.append(sentence)
        if len(sentence) > curr_max_length:
            curr_max_length = len(sentence)
        if i % batch_size == batch_size-1:
            batch_max_lengths.append(curr_max_length)
            curr_max_length = 0
    batch_max_lengths.append(curr_max_length)
#    print('BATCH MAX SENTENCE LENGTHS: {}'.format(batch_max_lengths))
    
    X_batches = []
    for batch_num in range(len(batch_max_lengths)):
        if batch_num == len(batch_max_lengths)-1:
            if include_last_batch:
                this_batch_size = len(training_data) % batch_size
            else:
                break
        else:
            this_batch_size = batch_size
        max_length = batch_max_lengths[batch_num]
        sentences_tensor = torch.zeros(this_batch_size, max_length).long()
        for sentence_num in range(this_batch_size):
            sentence = sentences[batch_num*batch_size+sentence_num]
            sentence_length = len(sentence)
            for word_index in range(sentence_length):
                sentences_tensor[sentence_num][max_length-sentence_length+word_index] = word_to_ix[sentence[word_index]]
        X_batches.append(sentences_tensor)
    return X_batches
    
def create_target_tensor_batches(training_data, tag_to_ix, batch_size, include_last_batch):
    targets = []
    for sentence, tag in training_data:
        targets.append(tag_to_ix[tag])
    targets_tensor = torch.LongTensor(targets)
    # Create batches.
    y_batches = []
    i = 0
    while batch_size*(i+1) <= len(training_data):
        y_batches.append(targets_tensor[batch_size*i:batch_size*(i+1)])
        i += 1
    if len(training_data) % batch_size != 0:
        y_batches.append(targets_tensor[len(training_data) - len(training_data) % batch_size:len(training_data)])
    return y_batches
        

# Reads the sentences and CoNLL files, tag_num should be 0-3.
# Outputs a list of tuples (sentence_as_list, tag).
def read_data(sentences_path, conll_path, tag_num):
    sentences = []
    tags = []
    conll_file = open(conll_path, "r")
    for line in conll_file:
        tags.append(line.split()[tag_num+1])
    sentences_file = open(sentences_path, "r")
    for line in sentences_file:
        if line.strip() != '':
            sentences.append(line.split())
    if len(sentences) != len(tags):
        print('Warning: sentences not same length as tags.')
    return list(zip(sentences, tags))


# Outputs a list of predictions, given the input batches.
def predict(X_batches, model, tag_to_ix):
    predictions = []
    ix_to_tag = dict((v, k) for k, v in tag_to_ix.items())
    for i in range(len(X_batches)):
        x = X_batches[i]
        if torch.cuda.is_available():
            x = x.cuda()
        tag_scores = model(x)
        _, max_indices = tag_scores.max(1)
        
        for j in range(max_indices.size()[0]):
            predictions.append(ix_to_tag[max_indices[j].item()])
    return predictions

def create_true_y(data, batch_size, include_last_batch):
    tags = []
    how_many_include = 0
    if include_last_batch:
        how_many_include = len(data)
    else:
        how_many_include = len(data) - len(data) % batch_size
    for i in range(how_many_include):
        tags.append(data[i][1])
    return tags


class LSTMModel(nn.Module):

    def __init__(self, embedding_dim_size, hidden_dim_size, num_hidden, vocab_size, tagset_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim_size)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim_size, hidden_dim_size, num_layers=num_hidden)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim_size*2, tagset_size)
        # Set up softmax, over the second dimension (because examples x output values).
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        if len(sentence.size()) == 1:
            sentence = sentence.view(1, -1)
        batch_size = sentence.size()[0]
        sentence_length = sentence.size()[1]
        # Put sentence in form (sentence_length, batch_size).
        sentence = sentence.permute(1,0)
        embeds = self.word_embeddings(sentence)
        # Input should be (sentence_length, batch_size, embedding_size).
        _, hidden_out = self.lstm(embeds)
        # Concatenate the hidden and cell states.
        hidden_out = torch.cat(hidden_out, dim=2)
        # Take the deepest hidden layer. Input should be (batch_size, embedding_size*2).
        # Output will be (batch_size, tag_set_size).
        tag_scores = self.hidden2tag(hidden_out[-1])
        tag_scores = self.softmax_layer(tag_scores)
        # Output is (batch_size, tag_set_size).
        return tag_scores

# Run an RNN experiment for a given tag and experiment.
# Tag num should be 0-3.
def run_rnn_experiment(tag_num, experiment_num):
    batch_size = BATCH_SIZE
    output_directory = 'drive/My Drive/Cog Sci Comps/RNN Experiments/experiment{}'.format(experiment_num)
    training_data = read_data("drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_dev-bpe.txt",
                              "drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_dev.txt", tag_num)
    dev_data = read_data("drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_train-filter8-bpe.txt",
                         "drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_train-filter8.txt", tag_num)
    test_data = read_data("drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test-bpe.txt",
                          "drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/phrase_tags_conll_test.txt", tag_num)
    # Don't use BPE encoded sentences for the test sentences path. That could mess up the sentence indices.
    test_sentences_path = "drive/My Drive/Cog Sci Comps/CoNLL-2012 Data/sentences_conll_test.txt"
    model_save_path = '{0}/rnn-model-tag{1}.pickle'.format(output_directory, tag_num+1)
    predictions_save_path = '{0}/predictions-rnn-model-tag{1}.txt'.format(output_directory, tag_num+1)
    observations_save_path = '{0}/observations-rnn-model-tag{1}.txt'.format(output_directory, tag_num+1)

    word_to_ix = {}
    tag_to_ix = {}
    # Use this to pad the sequence lengths.
    word_to_ix['XXXBLANKXXX'] = 0
    for sent, tag in training_data + dev_data + test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
        
    training_x = create_input_tensor_batches(training_data, word_to_ix, batch_size, False)
    training_y = create_target_tensor_batches(training_data, tag_to_ix, batch_size, False)
    dev_x = create_input_tensor_batches(dev_data, word_to_ix, batch_size, False)
    dev_y = create_target_tensor_batches(dev_data, tag_to_ix, batch_size, False)
    dev_true_y = create_true_y(dev_data, batch_size, False)
    test_x = create_input_tensor_batches(test_data, word_to_ix, batch_size, True)
    test_y = create_target_tensor_batches(test_data, tag_to_ix, batch_size, True)
    test_true_y = create_true_y(test_data, batch_size, True)

    print('Read all data.')
    
    # IF TRAIN IS TRUE:
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, NUM_HIDDEN, len(word_to_ix), len(tag_to_ix))
    if torch.cuda.is_available(): model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Run for the num of epochs or until 10 epochs with no improvement.
    best_acc = 0
    no_improvement = 0
    for epoch in range(MAX_EPOCHS):
        for i in range(len(training_x)):
            x = training_x[i]
            y = training_y[i]

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # Run the forward pass.
            tag_scores = model(x)

            optimizer.zero_grad()
            loss = loss_function(tag_scores, y)
            loss.backward()
            optimizer.step()

        # Evaluation.
        with torch.no_grad():
            print("Finished epoch {}: ".format(epoch))
            dev_y_hat = predict(dev_x, model, tag_to_ix)
        
        acc = accuracy(dev_y_hat, dev_true_y)
        print('Dev accuracy: {}'.format(acc))
        if acc > best_acc:
            no_improvement = 0
            best_acc = acc
            pickle.dump(model, open(model_save_path, "wb"))
            print('Saved to {}'.format(model_save_path))
        else:
            no_improvement += 1
        if no_improvement >= 10:
            print('Stopped training after {} epochs'.format(epoch+1))
            break
    print('Stopped training after the max of {} epochs'.format(MAX_EPOCHS))
    
    # Output the test results.
    best_model = pickle.load(open(model_save_path, "rb"))
    best_model.lstm.flatten_parameters()
    test_y_hat = predict(test_x, best_model, tag_to_ix)
    # Save predictions to the output predictions file.
    outfile = codecs.open(predictions_save_path, 'w', encoding='utf-8')
    for i in range(len(test_true_y)):
        outfile.write(test_y_hat[i])
        outfile.write('\t')
        outfile.write(test_true_y[i])
        outfile.write('\n')
    outfile.close()
    print('Saved predictions to {}'.format(predictions_save_path))
    # Save observations to the output observations file.
    test_sentence_indices = get_sentence_indices(test_sentences_path)
    sentence_accs = sentence_accuracies(test_y_hat, test_true_y, test_sentence_indices)
    outfile = codecs.open(observations_save_path, 'w', encoding='utf-8')
    for acc in sentence_accs:
        outfile.write(str(acc))
        outfile.write('\n')
    print('Saved observations to {}'.format(observations_save_path))
    
            
# ACTUALLY RUN THE EXPERIMENTS.
for experiment_num in range(4, 11):
    for tag_num in range(4):
        print('RUNNING EXPERIMENT {0} TAG {1}'.format(experiment_num, tag_num))
        run_rnn_experiment(tag_num, experiment_num)