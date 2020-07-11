# cog-sci-comps
Cognitive science comps: Emergence of generative syntax in neural machine translation models.

TODO: this code is generally not very clean. Maybe clean it up one day.

Outline of directories:
* conll2012_scripts: scripts to read and parse the CoNLL2012 dataset to extract constituent labels or create sentence trees. Can combine or filter CoNLL files.
* extract_data_representations: extract sentence representations from an existing OpenNMT-py model (assuming an LSTM RNN encoder). This code consists of modifications to the OpenNMT-py code, and it is not clean at all.
* lstm_rnn: run and evaluate a directly-trained RNN.
* opennmt_scripts: scripts to train an NMT model with OpenNMT-py. Can filter training data, parse training logs, and shuffle training sets with a random permutation. Includes Google Colab commands to run OpenNMT-py.
* stanford_parser: parse and evaluate the tree outputs of the Stanford parser.
* syntax_experiments: train feedforward NNs to predict constituent labels from vector representations. Includes scripts to compute baselines (MFT per word) and to evaluate model outputs.

See commands.txt for sample usages of various commands.