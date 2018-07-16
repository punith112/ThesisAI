import copy
import numpy as np

class perceptron:
	def __init__(self, genome = None, activation = None):
		if genome is not None:
			self.set_params(genome)
		self. activation = activation
		
	def get_params(self):
		return np.vstack([self.W, self.b])
	
	def set_params(self, params):
		self.W = params[:-1]
		self.b = params[-1]
		
	def compute(self, in_data):
		if self.activation:
			return self.activation(np.dot(in_data, self.W) + self.b)
		else:
			return np.maximum(np.dot(in_data, self.W) + self.b, 0)

class input_preprocessor:
	def __init__(self, mlp_genome):
		self.genome = mlp_genome
		if mlp_genome is not None:
			self.perceptrons = [perceptron(genome=gene) for gene in mlp_genome]
			self.perceptrons[-1].activation = np.tanh

	def get_genome(self):
		return(copy.deepcopy(self.genome))

	def set_genome(self, genome):
		if self.genome is None:
			self.perceptrons = [perceptron(genome=gene) for gene in genome]
			self.perceptrons[-1].activation = np.tanh
		else:
			for p, p_g in zip(self.perceptrons, genome):
				p.set_params(p_g)
		self.genome = genome
	
	def process(self, value):
		for per in self.perceptrons:
			value = per.compute(value)
		return value