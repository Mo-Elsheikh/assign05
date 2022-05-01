"""
For this assignment there is no automated testing. You will instead submit
your *.py file in Canvas. I will download and test your program from Canvas.
"""
import random
import math
import time
import sys
INF = sys.maxsize


def adjMatFromFile(filename):
    """ Create an adj/weight matrix from a file with verts, neighbors, and weights. """
    f = open(filename, "r")
    n_verts = int(f.readline())
    print(f" n_verts = {n_verts}")
    adjmat = [[None] * n_verts for i in range(n_verts)]
    for i in range(n_verts):
        adjmat[i][i] = 0
    for line in f:
        int_list = [int(i) for i in line.split()]
        vert = int_list.pop(0)
        assert len(int_list) % 2 == 0
        n_neighbors = len(int_list) // 2
        neighbors = [int_list[n] for n in range(0, len(int_list), 2)]
        distances = [int_list[d] for d in range(1, len(int_list), 2)]
        for i in range(n_neighbors):
            adjmat[vert][neighbors[i]] = distances[i]
    f.close()
    return adjmat


def TSPwGenAlgo(
        g,
        max_num_generations=5,
        population_size=10,
        mutation_rate=0.01,
        explore_rate=0.5
    ):
    """ A genetic algorithm to attempt to find an optimal solution to TSP  """
    solution_distance = None        # final solution distance
    solution = []                   # final path sequence of vertices found
    current_shortest_path = []      # A list to keep track of shortest path found in each generation
    number_of_cities = len(g)  
    discover_size = math.ceil(explore_rate * population_size)
    mate = int(population_size)
    population = generate_initial_population(population_size, number_of_cities) # create individual members of the population  
    for n in range(max_num_generations):
        for i in range(population_size):
            pop_result = (calc_distance_for_given_path(population[i][1], g), population[i][1])
            population[i] = pop_result
        population.sort(key=lambda city: city[0])
        current_shortest_path.append(population[0][0])
        children = []
        for i in range(population_size):
            parent_1 = mating(population[:discover_size], mate)
            parent_2 = mating(population[:discover_size], mate)
            children.append((0, crossover(parent_1[1], parent_2[1])))
        for swap_item1 in range(len(children)):  # Doing the mutation
            rand_val = random.random()
            if mutation_rate > rand_val:
                swap_item2 = random.randrange(number_of_cities)
                child1_temp = children[swap_item1]
                children[swap_item1] = children[swap_item2]
                children[swap_item2] = child1_temp
        population = children
  
    population.sort(key=lambda city: city[0])
    solution = population[0][1]
    solution_distance = calc_distance_for_given_path(population[0][1], g)
    solution = [population[0][1], solution[0]]
    return {
            'solution': solution,
            'solution_distance': solution_distance,
            'evolution': current_shortest_path
           }


def generate_initial_population(size, number_of_cities):
    """ Method to generate the initial population """
    initial_population = []
    for i in range(size):
        initial_population.append((0, random.sample(range(number_of_cities), number_of_cities)))
    return initial_population


def calc_distance_for_given_path(path, graph):
    """
    Method to get distance within a path
    """
    distance = 0
    for i in range(-1,(len(path) - 1), 1):
        distance += graph[path[i]][path[i + 1]]
    return distance


def crossover(parent1, parent2):
    """ Method to do the crossover """
    children = []
    child1 = []
    child2 = []
    
    first = int(random.random() * len(parent1))
    second = int(random.random() * len(parent1))
    # Selecting the start and end gene
    start_gene = first
    end_gene = second
    if (second < first):
        start_gene = first
        end_gene = first
    for i in range(start_gene, end_gene):
        child1.append(parent1[i])
    for element in parent2:
        if element not in child1:
            child2.append(element)
    children = child1 + child2
    return children


def mating(population, mating_size):
    """ Method to execute the Mating """
    mate_list = []

    for i in range(mating_size):
        pop_rand_val = population[random.randrange(len(population))]
        mate_list.append(pop_rand_val)
    #print (mate_list[0], mate_list[1], mate_list[-1])
    min_mate_val = mate_list[0]
    for mate in mate_list:
        if mate[0] < min_mate_val[0]:
            min_mate_val = mate
    return min_mate_val
   
def TSPwDynProg(g):
    """ (10pts extra credit) A dynamic programming approach to solve TSP """
    solution_path = [] # list of n+1 verts representing sequence of vertices with lowest total distance found
    solution_distance = INF # distance of solution path, note this should include edge back to starting vert

    #...

    return {
            'solution_path': solution_path,
            'solution_distance': solution_distance,
           }


def TSPwBandB(g):
    """ (10pts extra credit) A branch and bound approach to solve TSP """
    solution_path = [] # list of n+1 verts representing sequence of vertices with lowest total distance found
    solution_distance = INF # distance of solution path, note this should include edge back to starting vert

    #...

    return {
            'solution_path': solution_path,
            'solution_distance': solution_distance,
           }


def assign05_main():
    """ Load the graph (change the filename when you're ready to test larger ones) """
    g = adjMatFromFile("complete_graph_n08.txt")

    # Run genetic algorithm to find best solution possible
    start_time = time.time()
    res_ga = TSPwGenAlgo(g)
    elapsed_time_ga = time.time() - start_time
    print(f"GenAlgo runtime: {elapsed_time_ga:.2f}")
    print(f"  sol dist: {res_ga['solution_distance']}")
    print(f"  sol path: {res_ga['solution']}")

    # (Try to) run Dynamic Programming algorithm only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_dyn_prog = TSPwDynProg(g)
        elapsed_time = time.time() - start_time
        if len(res_dyn_prog['solution_path']) == len(g) + 1:
            print(f"Dyn Prog runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_dyn_prog['solution_distance']}")
            print(f"  sol path: {res_dyn_prog['solution']}")

    # (Try to) run Branch and Bound only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_bnb = TSPwBandB(g)
        elapsed_time = time.time() - start_time
        if len(res_bnb['solution_path']) == len(g) + 1:
            print(f"Branch & Bound runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_bnb['solution_distance']}")
            print(f"  sol path: {res_bnb['solution']}")


# Check if the program is being run directly (i.e. not being imported)
if __name__ == '__main__':
    assign05_main()
