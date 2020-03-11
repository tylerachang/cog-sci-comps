"""
LSTM RNN that takes sentence inputs (tokens separated by spaces) and outputs a classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    
    # REMOVE THIS: THIS IS JUST FOR THE TEST DATA
    while(len(idxs) < 4):
        idxs.insert(0, 0)
    
    return torch.tensor(idxs, dtype=torch.long)

# TODO: create batches here, with padding done by the max length per batch.
def create_input_tensor(training_data, word_to_ix):
    sentences = []
    max_length = 0
    for sentence, tag in training_data:
        sentences.append(sentence)
        if len(sentence) > max_length:
            max_length = len(sentence)
    sentences_tensor = torch.zeros(0, max_length).long()
    print('MAX SENTENCE LENGTH: {}'.format(max_length))
    
    # Faster to fill in the tensor instead of concatenating for each sentence.
    sentences_tensor = torch.zeros(len(sentences), max_length).long()
    for i in range(len(sentences)):
        sentence_length = len(sentences[i])
        for word_index in range(sentence_length):
            sentences_tensor[i][max_length-sentence_length+word_index] = word_to_ix[sentences[i][word_index]]
#    
#    for sentence in sentences:
#        idxs = [word_to_ix[w] for w in sentence]
#        # Pad the beginning of the sentences so that they are equal length.
#        while(len(idxs) < max_length):
#            idxs.insert(0, 0)
#        sentence_tensor = torch.tensor(idxs, dtype=torch.long)
#        sentence_tensor = sentence_tensor.view(1, max_length)
#        sentences_tensor = torch.cat(tuple([sentences_tensor, sentence_tensor]), dim=0)
    return sentences_tensor
    
def create_target_tensor(training_data, tag_to_ix):
    targets = []
    for sentence, tag in training_data:
        targets.append(tag_to_ix[tag])
    return torch.LongTensor(targets)
        

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


# ACTUALLY RUN THE RNN.
    
training_data = read_data("sentences_conll_test.txt", "phrase_tags_conll_test.txt", 0)
dev_data = read_data("sentences_conll_train-filter8.txt", "phrase_tags_conll_train-filter8.txt", 0)
training_data = training_data[0:10000]
dev_data = dev_data[0:100]

word_to_ix = {}
tag_to_ix = {}
# Use this to pad the sequence lengths.
word_to_ix['XXXBLANKXXX'] = 0
for sent, tag in training_data + dev_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)
        
training_x = create_input_tensor(training_data, word_to_ix)
training_y = create_target_tensor(training_data, tag_to_ix)
dev_x = create_input_tensor(dev_data, word_to_ix)
dev_y = create_target_tensor(dev_data, tag_to_ix)

EMBEDDING_DIM = 500
HIDDEN_DIM = 500
NUM_HIDDEN = 2
batch_size = 64
learning_rate = 0.0002

model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, NUM_HIDDEN, len(word_to_ix), len(tag_to_ix))
if torch.cuda.is_available(): model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create batches.
X_batches = []
y_batches = []
i = 0
while batch_size*(i+1) <= len(training_data):
    X_batches.append(training_x[batch_size*i:batch_size*(i+1), :])
    y_batches.append(training_y[batch_size*i:batch_size*(i+1)])
    i += 1
print('Created batches.')

for epoch in range(1000):
    for i in range(len(X_batches)):
        x = X_batches[i]
        y = y_batches[i]
        
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        
        # Run the forward pass.
        tag_scores = model(x)

        optimizer.zero_grad()
        loss = loss_function(tag_scores, y)
        loss.backward()
        optimizer.step()

    # Eval.
    with torch.no_grad():
        print("Finished epoch {}: ".format(epoch), end='')
        correct = 0
        for i in range(len(dev_data)):
            inputs = prepare_sequence(dev_data[i][0], word_to_ix)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            tag_scores = model(inputs)
            max_score, max_index = tag_scores.max(1)
            if tag_to_ix[dev_data[i][1]] == max_index[0].item():
                correct += 1
        print("accuracy {}".format(correct/len(dev_data)))
