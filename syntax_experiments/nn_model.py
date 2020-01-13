'''
Majority of this code taken from Deep RNNs Encode Soft Hierarchical Syntax (Blevins, Levy, & Zettlemoyer, 2018). Defines the feedforward NN model used in syntax experiments.
'''

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sklearn.metrics as metrics

from typing import Sequence, Union

#TODO: deal with CUDA usage more appropriately???
class Experiment_Model(torch.nn.Module):
	'''
	Class for the feed-forward network that is used 
	for embeddings feature classification experiments. 

	INPUTS
	num_classes: the number of classes on for this prediction experiment
	input_dim: size of the input representations
	num_layers: number of hidden layers in NN, defaults to 1
	hidden_dim: either an int (in which case all hidden layers have same dim) or 
		a list (which needs to be len = num_layers), defaults to input_dim
	dropout: either a float (in which case all layers have same dropout value) or
		a list (which needs to be len = num_layers), defaults to 0.0 

	PARAMETERS
	_input_dim: size of the input representations/ input dim for first hidden layer
	_output_dim: the output dimension of the last hidden layer
	_hidden_dims: the dimensions of each hidden layer
	_num_classes: number of classes (and dim of softmax layer, output of projection layer)
	_num_layers: the number of hidden layers
	_hidden_layers: torch.nn.Linear for each hidden layer 
	_dropout: torch.nn.Dropout for each hidden layer
	_activations: torch.nn.ReLU for each layer; forced to be ReLU,
		could be added as param later
	_projection_layer: a single torch.nn.Linear w/o activation that maps from dim of 
		last hidden layer to num_classes/dim of softmax
	_softmax_layer: torch.nn.Softmax

	'''

	def __init__(self, num_classes:int, input_dim: int, num_layers:int = 1, 
		hidden_dims: Union[int, Sequence[int]] = -1, 
		dropout: Union[float, Sequence[float]] = 0.0) -> None:
		super(Experiment_Model, self).__init__()

		#set up input dimensions
		self._input_dim = input_dim

		#set up number of layers
		self._num_layers = num_layers
		
		#set up hidden dimensions
		if not isinstance(hidden_dims, list):
			if hidden_dims == -1: hidden_dims = input_dim
			hidden_dims = [hidden_dims] * num_layers
		self._hidden_dims = hidden_dims

		#set up number of classes
		self._num_classes = num_classes

		#set up hidden layers
		in_dims = [input_dim] + hidden_dims[:-1]
		hidden_layers = []
		for in_dim, out_dim in zip(in_dims, hidden_dims):
			hidden_layers.append(torch.nn.Linear(in_dim, out_dim))
		self._hidden_layers = torch.nn.ModuleList(hidden_layers)

		#set up output dimensions
		self._output_dim = hidden_dims[-1]

		#set up dropout 
		if not isinstance(dropout, list): 
			dropout = [dropout] * num_layers
		dropout_by_layer = [torch.nn.Dropout(p=value) for value in dropout]
		self._dropout = torch.nn.ModuleList(dropout_by_layer)

		#set up activation -> forced to be ReLU here, could be changed later
		activation = torch.nn.ReLU()
		self._activations = [activation]*num_layers

		#set up projection layer
		self._projection_layer = torch.nn.Linear(self._output_dim, self._num_classes)

		#set up softmax
		self._softmax_layer = torch.nn.LogSoftmax()	

	'''
	Runs data through the model to get class probabilities of each datapoint
	'''
	def forward(self, input_data: torch.FloatTensor) -> torch.FloatTensor:
		data = input_data
		#pass data through hidden layers
		for layer, activation, dropout in zip(self._hidden_layers, self._activations, self._dropout):
			data = dropout(activation(layer(data)))
			#print('hidden layer res', data) #debugging

		#pass data through projection layer
		data = self._projection_layer(data)
		#print('projection res', data) #debugging

		#pass data through softmax
		data = self._softmax_layer(data)
		#print('softmax res', data) #debugging

		return data

'''
TODO: batching during training
'''
class Experiment():

	'''
	wrapper class for Experiment_Model

	INPUTS
	classes: list of classes in experiment/prediction task
	input_dim: for model init
	num_layers: for model init
	hidden_dim: for model init
	dropout: for model init
	
	PARAMETERS:
	_classes: list of classes
	_num_classes: number of classes
	model: the NN that acts as classifier for this experiment

	'''
	def __init__(self, classes: list, input_dim: int, num_layers:int = 1, 
		hidden_dims: Union[int, Sequence[int]] = -1, 
		dropout: Union[float, Sequence[float]] = 0.0) -> None:

		#classes set up
		num_classes = len(classes)
		self._classes = classes
		self._num_classes = num_classes

		#initalize model for experiment
		self.model = Experiment_Model(num_classes, input_dim, num_layers, hidden_dims, dropout)

		if torch.cuda.is_available(): self.model.cuda()

	'''
	turn y/label list into an n*1 tensor, where n = number of datapts 
	and each elm is the class index in the class list
	'''
	def y_to_tensor(self, labels) -> torch.LongTensor:
		#map labels to dimensions from classes index
		mapped_l = []
		for l in labels:
			mapped_l.append(self._classes.index(l))

		#should be a n*1 Tensor
		y = torch.LongTensor(mapped_l)

		return y

	'''
	turns input x (list of arrays) into a Tensor
	assumed to be list of row vector inputs
	'''
	def x_to_tensor(self, X):
		if isinstance(X, torch.Tensor): return X
		if isinstance(X, torch.cuda.FloatTensor): return X
		if isinstance(X, Variable): return X

		
		#TODO: deal with numpy

		X = torch.stack(X, dim=1)
		X = X.resize_(X.size()[1], X.size()[2])

		return X


	'''
	Trains the model on X_train (n*d), y_train (list of labels)
	'''
	def train(self, X_train: torch.FloatTensor, y_train: list, max_epochs:int = 10, learning_rate:float = 0.1) -> None:
		
		#checking input data dimensions against model
		X_train = self.x_to_tensor(X_train)
		if X_train.size()[1] != self.model._input_dim:
			raise Exception("input dataset should have the same feature dimension as the expected input dimension of the model.")
		
		#to long tensor of dim n*1
		y_train = self.y_to_tensor(y_train)

		#create training dataset and sampler
		#training_data = TensorDataset(X_train, y_train)
		#training_samples = DataLoader(training_data, batch_size=1000, pin_memory=True)

		#loss function currently set to NLL, could be parameterized later
		criterion = torch.nn.NLLLoss()

		#optimizer currently set to AdaGrad, could be parameterized later
		#lr = 0.01, should also be tuned
		optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)

		#Runs for either the num of epochs or if no improvement on dev set
		for epoch in range(0, max_epochs):

			#TODO: implement batching:: iterate through batches for each epoch
			#for batch_idx, (data, target) in enumerate(training_samples):
			
			data = X_train
			target = y_train

			if torch.cuda.is_available():
				data, target = data.cuda(), target.cuda()

			#initalize Variables for this epoch
			data, target = Variable(data), Variable(target)

			#forward pass
			target_hat = self.model(data)
			
			#compute loss
			loss = criterion(target_hat, target)

			#calc gradients and update weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print('finished epoch {}'.format(epoch))

		print('Stopped training after {} epochs'.format(max_epochs)) #DEBUGGING
		return

	'''
	Predicts labels from the model on a given dataset, X_eval (Tensor, n*d)
	TODO: add batching
	'''
	def predict(self, X_eval: torch.FloatTensor) -> list:
		#check dimensions of input data
		X_eval = self.x_to_tensor(X_eval)
		if X_eval.size()[1] != self.model._input_dim:
			raise AttributeError("input dataset should have the same dimensions as the expected input dimensions of the model.")
		
		#get class probabilities for each input in dataset
		if torch.cuda.is_available(): X_eval = X_eval.cuda()
		if not isinstance(X_eval, Variable): X_eval = Variable(X_eval)
		probs = self.model(X_eval)

		#predict a class for each input from these probabilties
		values, labels = probs.max(1)

		return [self._classes[l] for l in labels.data]


	'''
	Gives the performance metrics of the model on a given data/ labels pair
	Returns: accuracy of the model (1-misclassification error), f1 score, precision, recall
	'''
	def metrics(self, X_eval, y_eval):
		y_hat = self.predict(X_eval)
		n = ((float)(len(y_hat)))

		zipper = list(zip(y_eval, y_hat))

		accuracy = sum([1 for i, j in zipper if i == j])/n
		f1 = metrics.f1_score(y_hat, y_eval, average='micro')
		precision = metrics.precision_score(y_hat, y_eval, average='micro')
		recall = metrics.recall_score(y_hat, y_eval, average='micro')


		return accuracy, f1, precision, recall

	#Basically just a wrapper for sklearns classifcation report class, which allows us to break down the performance of the models on a class by class basis
	def classwise_report(self, X_eval, y_eval, filepath:str='', supress_print=True):
		y_hat = self.predict(X_eval)
		n = ((float)(len(y_hat)))

		cr = metrics.classification_report(y_eval, y_hat, target_names=[str(c) for c in self._classes], 
										digits=4)

		if not supress_print: print(cr)
		if len(filepath) > 0:
			f = open(filepath, 'w')
			f.write(cr)
			f.close()
		return


	def accuracy(self, X_eval, y_eval):
		y_hat = self.predict(X_eval)
		n = ((float)(len(y_hat)))
		zipper = list(zip(y_eval, y_hat))
		accuracy = sum([1 for i, j in zipper if i == j])/n
		return accuracy


