"""
Utility functions for syntax experiments.
"""

import torch

def load_reps(reps_path):
    reps = torch.load(reps_path)
    # Concatenate the last layer for hidden and context.
    # 0-3 increasing depth, 4-7 increasing depth.
    reps = torch.cat((reps[3, :, :], reps[7, :, :]), dim=1)
    return reps

def load_tags(tags_path, prediction_tag):
    tag_file = open(tags_path, "r")
    tag_lines = tag_file.readlines()
    y = []
    for line in tag_lines:
        tags = line.split()
        y.append(tags[prediction_tag])
    return y

def accuracy(y_hat, y_eval):
	n = ((float)(len(y_hat)))
	zipper = list(zip(y_eval, y_hat))
	accuracy = sum([1 for i, j in zipper if i == j])/n
	return accuracy

def sentence_averaged_accuracy(y_hat, y_eval, dev_sentence_indices):
	sentence_accs = sentence_accuracies(y_hat, y_eval, dev_sentence_indices)
	return sum(sentence_accs)/len(sentence_accs)

def sentence_accuracies(y_hat, y_eval, dev_sentence_indices):
	sentence_accs = []
	curr_correct = 0
	curr_count = 0
	for i in range(len(y_hat)):
		if i in dev_sentence_indices and i != 0:
			sentence_accs.append(curr_correct/curr_count)
			curr_correct = 0
			curr_count = 0
		if y_hat[i] == y_eval[i]:
			curr_correct += 1
		curr_count += 1
	sentence_accs.append(curr_correct/curr_count)
	return sentence_accs

def get_sentence_indices(sentences_path):
		sentence_indices = []
		sentences_file = open(sentences_path, "r")
		sentences = sentences_file.readlines()
		prev_sentence = 'XXXXXXXXXXXXXXXXXXXXX'
		for i in range(len(sentences)):
			if not prev_sentence.strip() in sentences[i]:
				sentence_indices.append(i)
			prev_sentence = sentences[i]
		return sentence_indices
	