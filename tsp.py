from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
# from data.tsp_data import distance_matrix
from data.tsp_data_2 import distance_matrix
# from data.tsp_data_3 import distance_matrix
from copy import deepcopy
from charles.selection import fps, tournament, rank
from charles.mutation import swap_mutation, inversion_mutation
from charles.crossover import cycle_co, pmx_co
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_fitness(self):
    """A simple objective function to calculate distances
    for the TSP problem.

    Returns:
        int: the total distance of the path
    """
    fitness = 0
    for i in range(len(self.representation)):
        fitness += distance_matrix[self.representation[i - 1]][self.representation[i]]
    return int(fitness)


def get_neighbours(self):
    """A neighbourhood function for the TSP problem. Switches
    indexes around in pairs.

    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation) - 1)]

    for count, i in enumerate(n):
        i[count], i[count + 1] = i[count + 1], i[count]

    n = [Individual(i) for i in n]
    return n


# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours


# pop = Population(
#     size=20,
#     sol_size=len(distance_matrix[0]),
#     valid_set=[i for i in range(len(distance_matrix[0]))],
#     replacement=False,
#     optim="min",
# )

# pop.evolve(
#     gens=20,
#     select=tournament,
#     crossover=pmx_co,
#     mutate=inversion_mutation,
#     co_p=0.9,
#     mu_p=0.1,
#     elitism=True
# )

# TODO - create a list of total generations to test
gens = [500]
# TODO add ranking selection to the list | create minimization for fps
selections = [fps, tournament, rank]
crossovers = [pmx_co, cycle_co]
mutations = [inversion_mutation, swap_mutation]
co_ps = [0.8, 0.9]
mu_ps = [0.1, 0.2]
elitism = [True, False]


# combination of all possible settings to the tsp problem
comb_settings = np.array(np.meshgrid(gens, selections, crossovers, mutations, co_ps, mu_ps, elitism)).T.reshape(-1, 7)


def evaluate(settings, runs=50, path='output/test50.csv'):
    df_final = pd.DataFrame(columns=['run', 'gens', 'select', 'crossover', 'mutate', \
                                     'co_p', 'mu_p', 'elitism', 'generation', 'fitness', 'time'])
    for run in range(1, runs + 1):
        for setting in settings:
            start = time.time()
            comb_set = f"gens: {setting[0]}; select: {str(setting[1]).split()[1]}; \
                        crossover: {str(setting[2]).split()[1]}; mutate: {str(setting[3]).split()[1]}; \
                        co_p: {setting[4]}; mu_p: {setting[5]}; elitism: {setting[6]}"
            pop = Population(
                size=20,
                sol_size=len(distance_matrix[0]),
                valid_set=[i for i in range(len(distance_matrix[0]))],
                replacement=False,
                optim="min",
            )
            list_gen, list_fitness = pop.evolve(
                                                gens=setting[0],
                                                select=setting[1],
                                                crossover=setting[2],
                                                mutate=setting[3],
                                                co_p=setting[4],
                                                mu_p=setting[5],
                                                elitism=setting[6]
            )
            df = pd.DataFrame()
            df['generation'] = list_gen
            df['fitness'] = list_fitness
            df['run'] = run
            df['gens'] = setting[0]
            df['select'] = str(setting[1]).split()[1]
            df['crossover'] = str(setting[2]).split()[1]
            df['mutate'] = str(setting[3]).split()[1]
            df['co_p']=setting[4]
            df['mu_p'] = setting[5]
            df['elitism'] = setting[6]
            df['time'] = time.time() - start
            df = df[['run', 'gens', 'select', 'crossover', 'mutate', 'co_p', 'mu_p', 'elitism', 'generation', 'fitness', 'time']]
            #df_final = pd.concat([df_final, df])
            df_final = df_final.append(df)

    df_final.to_csv(path, sep=';', index=False)

    return print('Done - ' + str(df_final['time'].sum())) 

evaluate(settings=comb_settings, runs=50)


# plot all combinations
df_plot = pd.read_csv(path, sep=';')
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.lineplot(data=df_plot, x='generation', y='fitness', hue='settings', estimator='mean', ci=99, err_style='band', ax=ax, legend=True)
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0)
plt.show()

