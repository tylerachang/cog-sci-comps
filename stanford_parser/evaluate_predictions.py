# Copy utils.py from the syntax_experiments directory.
from utils import *
import codecs

for i in range(1,5):
    y, y_hat = read_predictions('predictions-PCFGSentenceTrainedParser-tag{}.txt'.format(i))
    
    sentence_accs = sentence_accuracies(y_hat, y, get_sentence_indices('sentences_conll_test.txt'))
    outfile = codecs.open('observations-PCFGSentenceTrainedParser-tag{}.txt'.format(i), 'w')
    for acc in sentence_accs:
        outfile.write('{}\n'.format(acc))
    # Print regular (non-sentence-averaged accuracies).
    print(accuracy(y_hat, y))