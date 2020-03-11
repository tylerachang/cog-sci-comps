"""
LSTM RNN that takes sentence inputs (tokens separated by spaces) and outputs a classification.
TODO: process in batches instead of sentence by sentence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def one_hot(tag, tag_to_ix):
    target = [tag_to_ix[tag]]
    return torch.LongTensor(target)

# Reads the sentences and CoNLL files, tag_num should be 0-3.
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
    return list(zip(sentences, tags))

class LSTMModel(nn.Module):

    def __init__(self, embedding_dim_size, hidden_dim_size, vocab_size, tagset_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim_size, hidden_dim_size, num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim_size*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        _, hidden_out = self.lstm(embeds.view(len(sentence), 1, -1))
        hidden_out = torch.cat(hidden_out, dim=2)        
        tag_scores = self.hidden2tag(hidden_out)        
        softmax_tag_scores = F.softmax(tag_scores, dim=2)
        return softmax_tag_scores.view(1,-1)


training_data = read_data("sentences_conll_test.txt", "phrase_tags_conll_test.txt", 0)
dev_data = read_data("phrase_tags_conll_train-filter8.txt", "phrase_tags_conll_train-filter8.txt", 0)
dev_data = dev_data[0:100]

word_to_ix = {}
tag_to_ix = {}
for sent, tag in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)
for sent, tag in dev_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    if tag not in tag_to_ix:
        tag_to_ix[tag] = len(tag_to_ix)

EMBEDDING_DIM = 500
HIDDEN_DIM = 500

model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()
    
    example_counter = 0
    for sentence, tag in training_data:
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        target = one_hot(tag, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, target)
        loss.backward()
        example_counter += 1
        
        if example_counter % 100 == 0:
            optimizer.step()
            model.zero_grad()
        
        if example_counter % 1000 == 0:
            # Eval.
            with torch.no_grad():
                correct = 0
                for i in range(len(dev_data)):
                    inputs = prepare_sequence(dev_data[i][0], word_to_ix)
                    tag_scores = model(inputs)
                    max_score, max_index = tag_scores.max(1)
                    if tag_to_ix[dev_data[i][1]] == max_index[0].item():
                        correct += 1
                print("Finished epoch {0}: accuracy {1}".format(epoch, correct/len(dev_data)))
