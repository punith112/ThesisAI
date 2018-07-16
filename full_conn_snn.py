import numpy as np
import scipy.stats as ss
import copy

from snn_preprocessing import *


#########
# UTILS #
#########

def random_truncnorm(loc, scale, low, high, sample_size = 1):
	a, b = (low - loc) / scale, (high - loc) / scale
	sample = [ss.truncnorm.rvs(a, b, loc, scale) for _ in range(sample_size)]
	return np.array(sample)

def create_weight(in_size, out_size = None, sigma = .5):
	if out_size is None:
		out_size = in_size
	return np.random.normal(loc=0, scale= sigma, size=[out_size, in_size])


##########
# AGENTS #
##########

class snn_layer:
	def __init__(self, size, activation_func = lambda x: (np.tanh(x)+1)/2 ):
		self.size = size
		self.a= random_truncnorm(loc=.8, scale=.5, low = 0, high = 1, sample_size= self.size)
		self.b= random_truncnorm(loc=.7, scale=.5, low = 0, high = 1, sample_size= self.size)
		self.c= random_truncnorm(loc=.5, scale=.1, low = 0, high = 1, sample_size= self.size)
		self.reset()
		
		self.activation_func = activation_func
		
	def reset(self):
		self.potentials = np.zeros(self.size)
		self.thresholds = copy.copy(self.c)
		self.spikes = np.zeros(self.size)
	
	def apply_input(self, in_current):
		self.potentials += in_current

	def update_state(self):
		# deterministic spikes
		self.spikes = self.thresholds < self.potentials
		
		# stochastic spikes
		#spike_p = np.minimum(1, self.activation_func(self.potentials - self.thresholds))
		#self.spikes = np.random.binomial(n = 1, p = spike_p) == 1
		
		# spiked neurons potentials reset and treshold raises
		self.thresholds[self.spikes] += (self.b*self.potentials)[self.spikes]
		self.potentials[self.spikes] = 0
		# non spiking neurons decay
		self.potentials[~self.spikes] *= self.a[~self.spikes]
		self.thresholds[~self.spikes] += (self.b*self.potentials)[~self.spikes]
		
		self.thresholds += (self.c - self.thresholds)*self.b/2
		
	def get_genome(self):
		return np.vstack([self.a, self.b, self.c])
		
	def set_genome(self, genome):
		a, b, c = genome
		self.a = np.maximum(np.minimum(a, 1), 0.001)
		self.b = np.maximum(np.minimum(b, 1), 0.001)
		self.c = np.minimum(c, 0.001)
		self.reset()


class fully_connected_snn:
	def __init__(self, input_size, prepro_hidden = [64, 64], preprocessed_size = 32,
				 status_layer_size = 16, output_layer_size = 2, symmetric_weights = True,
				spike_train_len = 20):
		
		#input preprocessor
		prepro_sizes = [input_size] + prepro_hidden + [preprocessed_size]
		prepro_genome = [np.random.normal(loc= 0, scale= .5, size=[prepro_sizes[i-1] + 1, prepro_sizes[i]])
					 for i in range(1,len(prepro_sizes))]
		self.prepro = input_preprocessor(prepro_genome)
		
		#spiking layers
		self.cl = snn_layer(preprocessed_size)
		self.sl = snn_layer(status_layer_size)
		self.ol = snn_layer(output_layer_size)
		
		
		self.weights = {}
		# create input weight
		self.weights[self.cl] = np.random.normal(loc=0, scale= .5, size=(preprocessed_size))
		# connect contest layer to the first status one
		self.weights[(self.cl, self.sl)] = create_weight(preprocessed_size, status_layer_size)
		# connect back and forth the status layers with the outputs, optionally with symmetric weights
		w = create_weight(status_layer_size, output_layer_size)
		self.weights[(self.sl, self.ol)] = w
		self.weights[(self.ol, self.sl)] = w.T if symmetric_weights else create_weight(output_layer_size,
																					   status_layers_size)
		# create recurrent weights
		self.weights[self.ol] = create_weight(output_layer_size) # for output layer
		self.weights[self.sl] = create_weight(status_layer_size) # and for the status
		
		# the output is based on frequency
		self.spike_train = np.zeros([output_layer_size, spike_train_len])
		self.sin_genome = np.random.poisson(lam = 2, size = output_layer_size)
		
	def update_network(self):
		self.cl.update_state()
		self.sl.update_state()
		self.ol.update_state()

	def reset_network(self):
		self.cl.reset()
		self.sl.reset()
		self.ol.reset()
		self.spike_train = np.zeros(self.spike_train.shape)
	
	def transmit_spikes(self, layer0, layer1 = None):
		if layer1 is None:
			w = self.weights[layer0]
			layer0.apply_input(np.dot(w, layer0.spikes))
		else:
			w = self.weights[(layer0, layer1)]
			layer1.apply_input(np.dot(w, layer0.spikes))
	
	def act(self, input_data, intensity = 1):
		data = self.prepro.process(input_data)
		self.cl.apply_input(self.weights[self.cl] * data)
		
		self.transmit_spikes(self.cl, self.sl)
		self.transmit_spikes(self.sl)
		self.transmit_spikes(self.sl, self.ol)
		self.transmit_spikes(self.ol, self.sl)
		self.transmit_spikes(self.ol)
		#self.print_state() 
		self.update_network()
		
		
		self.spike_train[:,:-1] = self.spike_train[:, 1:]
		self.spike_train[:,-1] = self.ol.spikes
		
		freq = 1-np.mean(self.spike_train, axis = 1)
		freq = np.multiply(freq, self.sin_genome)
		return np.multiply(np.sinc(freq), intensity)
		
		
	def print_state(self):
		print('\t | \t Potentials: \t | \t Thresholds: \t | \t Spikes:')
		print(' CL \t | {} | {} | {}'.format(self.cl.potentials, self.cl.thresholds, self.cl. spikes))
		print(' SL \t | {} | {} | {}'.format(self.sl.potentials, self.sl.thresholds, self.sl. spikes))
		print(' OL \t | {} | {} | {}'.format(self.ol.potentials, self.ol.thresholds, self.ol. spikes))
		print()


class snn_individual(fully_connected_snn):
	def __init__(self, input_size, prepro_hidden = [64, 64], preprocessed_size = 32,
				 status_layer_size = 16, output_layer_size = 2, symmetric_weights = True,
				spike_train_len = 20):
		
		fully_connected_snn.__init__(self, input_size, prepro_hidden, preprocessed_size,
				 status_layer_size, output_layer_size, symmetric_weights, spike_train_len)
		
		self.fitness = 0
		
	def get_genome(self):
		return [self.prepro.get_genome(), self.cl.get_genome(), self.sl.get_genome(),
				self.ol.get_genome(), list(self.weights.values()), self.sin_genome]
	
	def set_genome(self, genome):
		prepro_gen, cl_gen, sl_gen, ol_gen, weights_gen, sin_gen = genome
		self.prepro.set_genome(prepro_gen)
		self.cl.set_genome(cl_gen)
		self.sl.set_genome(sl_gen)
		self.ol.set_genome(ol_gen)
		self.sin_genome = sin_gen
		
		for k, w_gen in zip(self.weights, weights_gen):
			self.weights[k] = w_gen
			
	def evaluate(self, fit_fun):
		self.fitness = fit_fun(self)


class snn_population:
	def __init__(self, pop_size = 20, input_size = 4,
				 prepro_hidden = [16, 16], preprocessed_size = 4, status_layer_size = 16,
				 output_layer_size = 2, symmetric_weights = True, spike_train_len = 20,
				 fitness_function = None):

		self.fitness_function = fitness_function

		self.individuals = [snn_individual(input_size, prepro_hidden, preprocessed_size,
										  status_layer_size, output_layer_size,
										  symmetric_weights, spike_train_len)
						   for _ in range(pop_size)]

		self.best_list = []
		self.best_score = 0
		self.evaluate_pop(fitness_function)
		self.find_best()

	def get_individuals(self):
		return(self.individuals)

	def remove_individual(self, ind):
		self.individuals.remove(ind)

	def add_individual(self, new_ind):
		self.individuals.append(new_ind)

	# evolutionary stuff
	def evaluate_pop(self, fitness_function):
		if fitness_function:
			for i in self.individuals:
				i.evaluate(fitness_function)
		else:
			for i in self.individuals:
				i.evaluate(self.fitness_function)

	def find_best(self):
		scores = [i.fitness for i in self.individuals]
		self.best_score = np.max(scores)
		self.best = self.individuals[np.argmax(scores)]
		self.average_score = np.mean(scores)
		self.best_list.append(self.best_score)

