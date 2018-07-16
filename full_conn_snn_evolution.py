import numpy as np
import random
import copy
import scipy.stats as ss


from full_conn_snn import *

############
#### DE ####
############

def select_parents(population, pairs = 1, b = True):
	popu = population.get_individuals()
	if b:
		parents = list(random.sample(popu, k= pairs*2)) + [population.best]
	else:
		parents = list(random.sample(popu, k= pairs*2 +1))
	return parents

def mutate_DE(original_gen, base_gen, extra_gens, f = 0.5, cp = 0.4):
	if type(original_gen) == list:
		new_gen = [mutate_DE(or_gen, base_gen[g], [e_g[g] for e_g in extra_gens], f, cp) for g, or_gen in enumerate(original_gen)]
	else:
		new_gen = base_gen + sum([f*gene*(-1)**i for i, gene in enumerate(extra_gens)])
		chance = np.random.binomial(n = 1, p= cp, size = new_gen.shape)    
		new_gen = chance*new_gen + (1-chance)*original_gen
	return new_gen

def create_mutant(original, parents, f = 0.5, cp = 0.4):
	original_genome = original.get_genome()
	base_genome = parents[-1].get_genome()
	extra_genomes = [par.get_genome() for par in parents[:-1]]

	new_genome = mutate_DE(original_genome, base_genome, extra_genomes, f, cp)
	
	return new_genome


def DE(population, fitness_function, f = 0.5, cp = 0.4, np = 1, b = True):
	new_pop = []
	for original in population.get_individuals():
		parents = select_parents(population, np, b)
		child = copy.deepcopy(original)
		child.set_genome(create_mutant(original, parents, f, cp))
		child.evaluate(fitness_function)

		if child.fitness > original.fitness:
			new_pop.append(child)
		else:
			new_pop.append(original)

	population.individuals = new_pop
	population.find_best()

	return population


#############
#### NES ####
#############

def perturbate_genome(original_gen, sigma = .25):
	if type(original_gen) == list:
		new_gen1 =[]
		new_gen2 =[]
		e1 = []
		e2 = []
		for or_gen in original_gen:
			n_g1, e_1, n_g2, e_2 = perturbate_genome(original_gen, sigma)
			new_gen1.append(n_g1)
			new_gen2.append(n_g2)
			e1.append(e_1)
			e2.append(e_2)
	else:
		e1 = np.random.normal(scale = sigma, size = original_gen.shape)
		e2 = -e1
		new_gen1 = original_gen + e1
		new_gen2 = original_gen + e2

	return new_gen1, e1, new_gen2, e2


def perturbate_actor(original, sigma = .25):
	original_gen = original.get_genome()
	new_gen1, epsilon1, new_gen2, epsilon2 = perturbate_genome(original_genome, sigma)
	new_actor1 = copy.deepcopy(original)
	new_actor2 = copy.deepcopy(original)
	new_actor1.set_genome(new_gen1)
	new_actor2.set_genome(new_gen2)

	return epsilon1, new_actor1, epsilon2, new_actor2

def combine_genomes(original_gen, actors_fitness, epsilons, learning_rate = .01, F0 = 0):
	if type(original_gen) == list:
		mutation = [combine_genomes(or_gen, actors_fitness, [eps[i] for eps in epsilons], learning_rate, F0) for i, or_gen in enumerate(original_gen)]
	else:
		mutation = original_gen
		n = len(actors_fitness)
		for F, epsilon_i in zip(actors_fitness, epsilons):
			mutation += learning_rate*(epsilon_i*(F-F0))/n

	return mutation

def combine_actors(original, actors, epsilons, learning_rate = .01, relative = False):
	F0 = original.fitness if relative else 0
	new_actor = copy.deepcopy(original)
	new_actor.set_genome(combine_genomes(original.get_genome(), [a.fitness for a in actors], epsilons, learning_rate, F0))

	return new_actor

def OLD_perturbate_actor(original, sigma = .25):
	new_actor1 = copy.deepcopy(original)
	new_actor2 = copy.deepcopy(original)

	genome1 = new_actor1.get_genome()
	genome2 = new_actor2.get_genome()

	epsilon1 = []
	epsilon2 = []
	for i_gb, gene_block in enumerate(genome1):
		e1 = []
		e2 = []
		for i_g, gene in enumerate(gene_block):
			# generate an error for each block of parameters (layers' weights and biases)
			e = np.random.normal(scale = sigma, size = gene.shape)

			# apply the error in symmetric fashion
			genome1[i_gb][i_g] += e
			genome2[i_gb][i_g] -= e

			e1.append(e/sigma)
			e2.append(-e/sigma)
		epsilon1.append(e1)
		epsilon2.append(e2)

	new_actor1.set_genome(genome1)
	new_actor2.set_genome(genome2)

	return epsilon1, new_actor1, epsilon2, new_actor2


def OLD_combine_actors(original, actors, epsilons, learning_rate = .01, relative = False):
	mutation = original.get_genome()
	F0 = original.fitness if relative else 0
	n = len(actors)
	for i, actor in enumerate(actors):
		F = actor.fitness
		epsilon_i = epsilons[i]
		for i_gb, gene_block in enumerate(mutation):
			for i_g, gene in enumerate(gene_block):
				gene += learning_rate*(epsilon_i[i_gb][i_g]*(F-F0))/n

	new_actor = copy.deepcopy(original)
	new_actor.set_genome(mutation)

	return new_actor

def NES(actor, fitness_function, sample_size = 20, sigma = .25, learning_rate = .01, relative = False):

	sample = []
	epsilons = []
	for _ in range(sample_size):
		e1, a1, e2, a2 = perturbate_actor(actor, sigma)
		sample += [a1, a2]
		epsilons += [e1, e2]

	for s in sample:
		s.evaluate(fitness_function)

	new_actor = combine_actors(actor, sample, epsilons, learning_rate, relative)

	new_actor.evaluate(fitness_function)

	return new_actor

#########
## GA  ##
#########

def select_parents_fit_rank(population, parents_num, rank = True, relative = False):
    individuals = population.get_individuals()
    if rank:
        weights = np.array([ind.fitness for ind in individuals])
        weights -= np.min(weights)
        if relative:
            weights = ss.rankdata(-weights)
        weights /= sum(weights)

        parents = list(np.random.choice(individuals, size = parents_num, p = weights))
    else:
        parents = list(np.random.choice(individuals, size = parents_num))

    return parents

def mutate_gauss(original_genome, sigma = .05):
    if type(original_genome) == list:
        new_gen = [mutate_gauss(or_gen, sigma) for or_gen in original_genome]
    else:
        new_gen = original_genome + np.random.normal(loc= 0, scale= sigma, size = original_genome.shape)

    return new_gen

def GA(population, fitness_function = None, sigma = .05, elite_num = 1, rank = True, relative = False):
    individuals = population.get_individuals()
    pop_fitness = [ind.fitness for ind in individuals]
    elites = copy.deepcopy([individuals[i] for i in np.argpartition(pop_fitness, -elite_num)[-elite_num:] ])
    parents = select_parents_fit_rank(population, len(individuals) - elite_num, rank, relative)
    new_pop = []
    for ind in parents:
        child = copy.deepcopy(ind)
        child.set_genome(mutate_gauss(ind.get_genome(), sigma))
        new_pop.append(child)

    population.individuals = new_pop
    population.evaluate_pop(fitness_function)
    for elite in elites:
        population.add_individual(elite)
    population.find_best()

    return population