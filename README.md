# Project CIFO - COMPUTATIONAL INTELLIGENCE FOR OPTIMIZATION

Genetic Algorithm applied to the TSP problem.<br>
### Settings developed:
- Selection:
  - tournament
  - rank
  - fps (fitness proportionate selection)
- Crossover:
  - cycle_co (cycle crossover)
  - pmx_co (partially matched crossover)
- Mutation:
  - swap_mutation
  - inversion_mutation

___

### Code organization:
- [tsp.py](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/tsp.py): file that runs the genetic algorithm for the TSP problem using combination of selected setting
- [charles](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/tree/main/charles): folder that contains all the codes for the Genetic Algorithm 
- [charles.py](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/charles/charles.py): definitions of the Individual and Population classes
- [crossover.py](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/charles/crossover.py): functions for the crossover implementations
- [mutation.py](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/charles/mutation.py): functions for the mutation implementations
- [selection.py](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/charles/selection.py): functions for the selection implementations
- [data](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/tree/main/data): distances matrix used for implementing TSP problems:
  - tsp_data.py: distance matrix of 13 cities
  - tsp_data_2.py: distance matrix of 29 cities
  - tsp_data_3.py: distance matrix of 561 cities
- [Analysis_TSP.ipynb](https://github.com/anabeatrizfig/Comp_Intelligence_Optimization/blob/main/Analysis_TSP.ipynb): Jupyter Notebook with the plots of the Genetic Algorithms processes
- the other folder are from old implementations, and it is not used for the TSP problem
