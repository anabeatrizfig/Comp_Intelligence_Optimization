from charles.charles import Population, Individual
# from data.tsp_data import distance_matrix
from data.tsp_data_2 import distance_matrix
# from data.tsp_data_3 import distance_matrix
from copy import deepcopy
from charles.selection import fps, tournament, rank
from charles.mutation import swap_mutation, inversion_mutation
from charles.crossover import cycle_co, pmx_co
import pandas as pd
import numpy as np
import time


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

# select the combination of settings to run
gens = [500]
selections = [tournament]  # fps, rank]
crossovers = [pmx_co]  # , cycle_co]
mutations = [inversion_mutation]  # , swap_mutation]
co_ps = [0.9]  # , 0.8]
mu_ps = [0.1]  # , 0.2]
elitism = [True]  # , False]

# combination of all possible settings to the tsp problem
comb_settings = np.array(np.meshgrid(gens, selections, crossovers, mutations, co_ps, mu_ps, elitism)).T.reshape(-1, 7)


def evaluate(settings, runs=50, path='output/test50.csv'):
    """
    An evaluation function to test different parameters for the Genetic Algorithms
    and compute the best fitness and the processing time

     Args:
        settings: a list of combination of the parameters
        runs: number of runs for each setting
        path: the location folder to save the dataset with the results

    Returns:
        print of the success of the code and the time spend
    """
    # Create an empty dataframe to store the values
    df_final = pd.DataFrame(columns=['run', 'gens', 'select', 'crossover', 'mutate',
                                     'co_p', 'mu_p', 'elitism', 'generation', 'fitness', 'time'])
    # Loop into all the combinations in the number of runs defined by the function
    for run in range(1, runs + 1):
        for setting in settings:
            # store the time of the beginning of each setting
            start = time.time()

            comb_set = f"""
            (gens: {setting[0]}; select: {str(setting[1]).split()[1]};
            crossover: {str(setting[2]).split()[1]}; mutate: {str(setting[3]).split()[1]};
            co_p: {setting[4]}; mu_p: {setting[5]}; elitism: {setting[6]})"""

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
            # insert each best feature of each generation for the settings
            df['generation'] = list_gen
            df['fitness'] = list_fitness
            df['run'] = run
            # insert all the setting used in this run and the time spent
            df['gens'] = setting[0]
            df['select'] = str(setting[1]).split()[1]
            df['crossover'] = str(setting[2]).split()[1]
            df['mutate'] = str(setting[3]).split()[1]
            df['co_p'] = setting[4]
            df['mu_p'] = setting[5]
            df['elitism'] = setting[6]
            df['time'] = time.time() - start
            df = df[['run', 'gens', 'select', 'crossover', 'mutate', 'co_p', 'mu_p', 'elitism', 'generation', 'fitness',
                     'time']]
            # append this dataframe to the final one
            df_final = pd.concat([df_final, df])
    # save the dataframe in the folder
    df_final.to_csv(path, sep=';', index=False)

    return print('Done - ' + str(round(df_final['time'].sum(), 2)) + 'ms', f'{comb_set}')


folder = 'output/test50.csv'
evaluate(settings=comb_settings, runs=50, path=folder)
