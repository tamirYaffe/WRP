import random, copy, statistics, time, csv
from tkinter import *
from deap import base
from deap import creator
from deap import tools
import networkx as nx
import numpy as np
from bresenham import bresenham
import queue as Q

# define the maze.
from GUI import CellGrid

toolbox = base.Toolbox()
G = nx.Graph()
all_seen_size = 0
all_shortest_paths = []
seen_dictionary = {}
random.seed(128)


# define point in the maze.
class Position:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x=0, y=0):
        """ Create a new point at the origin """
        self.x = x
        self.y = y

    def is_exsits_los(self, other):
        path = bresenham(self.x, self.y, other.x, other.y)
        exist = True
        for point in path:
            if maze[point[1]][point[0]] == 1:
                exist = False
                break
        return exist

    def get_seen_list(self):
        seen = set()
        for i in range(0, len(maze)):
            for j in range(0, len(maze[0])):
                other = Position(j, i)
                if maze[i][j] == 1 or self == other:
                    continue
                if self.is_exsits_los(other) or other.is_exsits_los(self):
                    seen.add(other)
        return seen

    def __eq__(self, other):
        return self.x is other.x and self.y is other.y

    def __lt__(self, other):
        return len(seen_dictionary[self]) < len(seen_dictionary[other])

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str(self.x) + "," + str(self.y)


maze = [[1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]]

# define start position
# entry_position = Position(6, 9)
entry_position = Position(0, 0)
# pivots = [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]
pivots = []




def two_mutate_random(individual):
    num = random.random()
    if num < 0.5:
        onePointMutate(individual)
    else:
        swapMutate(individual)


def onePointMutate(individual):
    index = random.randrange(0, len(individual))
    pivot = individual[index][0]
    pivot_watchers = list(seen_dictionary[pivot])
    watcher = pivot_watchers[random.randrange(0, len(pivot_watchers))]
    while individual[index][1] is watcher:
        watcher = pivot_watchers[random.randrange(0, len(pivot_watchers))]
    individual[index] = (pivot, watcher)


def swapMutate(individual):
    index1 = random.randrange(0, len(individual))
    index2 = random.randrange(0, len(individual))
    while index1 is index2:
        index2 = random.randrange(0, len(individual))
    temp = individual[index2]
    individual[index2] = individual[index1]
    individual[index1] = temp

def singlePointCrossoverTwoChilds(individual1, individual2):
    return singlePointCrossover(individual1, individual2), singlePointCrossover(individual2, individual1)


def singlePointCrossover(individual1, individual2):
    index = random.randrange(0, len(individual1))
    child = []
    child_pivots = []
    for i in range(0, len(individual2)):
        if i < index:
            child.append(individual1[i])
            child_pivots.append(individual1[i][0])
        else:
            if individual2[i][0] not in child_pivots:
                child.append(individual2[i])
                child_pivots.append(individual2[i][0])
            else:
                if len(individual1) > i:
                    child.append(individual1[i])
                    child_pivots.append(individual1[i][0])
                else:
                    for j in range(0, index):
                        if individual2[j][0] not in child_pivots:
                            child.append(individual2[j])
                            child_pivots.append(individual2[j][0])
    if len(child) < len(individual1):
        for attr in individual1:
            if attr[0] not in child_pivots:
                child.append(attr)
    return child


def add_pivots_to_individual(seen_set, individual):
    pivots_to_add = []
    node_queue = Q.PriorityQueue()
    pivot_nodes = set()
    for node in set(G.nodes) - seen_set:
        node_queue.put(node)
    while not node_queue.empty():
        node = node_queue.get()
        if node not in pivot_nodes and len(pivot_nodes & seen_dictionary[node]) == 0:
            pivots_to_add.append(node)
            pivot_nodes.add(node)
            pivot_nodes.update(seen_dictionary[node])
    for pivot in pivots_to_add:
        min_dist_pivot = 1000000
        min_dist_pivot_index = 0
        index = 0
        for attr in individual:
            individual_pivot = attr[0]
            distance = len(all_shortest_paths[pivot][individual_pivot])
            if distance < min_dist_pivot:
                min_dist_pivot = distance
                min_dist_pivot_index = index
            index = index+1
        index = min_dist_pivot_index + 1
        pivot_watchers = list(seen_dictionary[pivot])
        random_watcher = pivot_watchers[random.randrange(0, len(pivot_watchers))]
        attr_to_add = (pivot, random_watcher)
        while index < len(individual):
            temp_attr = individual[index]
            individual[index] = attr_to_add
            attr_to_add = temp_attr
            index = index + 1
        individual.append(attr_to_add)


def eval(individual):
    global seen_all
    seen_set = set()
    seen_set.update(seen_dictionary[entry_position])
    current_position = Position(entry_position.x, entry_position.y)
    move_ctr = 0
    path_positions = []
    path_positions.append(current_position)
    for watcher in individual:
        path = all_shortest_paths[current_position][watcher[1]][1:]
        for position in path:
            seen_set.update(seen_dictionary[position])
        path_positions.extend(path)
        move_ctr = move_ctr + len(path)
        if len(path) > 0:
            current_position = path[-1]
    if len(seen_set) != all_seen_size:
        add_pivots_to_individual(seen_set, individual)
        return eval(individual)
    return move_ctr,


def print_path(best_ind):
    current_position = Position(entry_position.x, entry_position.y)
    path_positions = []
    path_positions.append(current_position)
    for watcher in best_ind:
        path = all_shortest_paths[current_position][watcher[1]][1:]
        path_positions.extend(path)
        if len(path) > 0:
            current_position = path[-1]
    for position in path_positions:
        maze[position.y][position.x] = 2
    root = Tk()
    CellGrid(root, len(maze), len(maze[0]), 40, maze)
    root.mainloop()


def pre_processing():
    global all_seen_size
    global G
    global all_shortest_paths, seen_dictionary

    # create seen dictionary according to the maze.
    for i in range(0, len(maze)):
        for j in range(0, len(maze[0])):
            if maze[i][j] == 1:
                continue
            position = Position(j, i)
            seen_dictionary[position] = position.get_seen_list()

    # make graph from maze
    for i in range(0, len(maze)):
        for j in range(0, len(maze[0])):
            if maze[i][j] == 0:
                all_seen_size = all_seen_size + 1
                G.add_node(Position(j, i))
                if j != 0 and maze[i][j - 1] != 1:
                    G.add_edge(Position(j, i), Position(j - 1, i), weight=1)
                if i != 0 and maze[i - 1][j] != 1:
                    G.add_edge(Position(j, i), Position(j, i - 1), weight=1)
    # build all shortest dict
    all_shortest_paths = dict(nx.all_pairs_shortest_path(G))
    # extract pivots
    node_queue = Q.PriorityQueue()
    pivot_nodes = set()
    for node in G.nodes:
        # check not in seen of start
        if node not in seen_dictionary[entry_position]:
            node_queue.put(node)
    while not node_queue.empty():
        node = node_queue.get()
        if node not in pivot_nodes and len(pivot_nodes & seen_dictionary[node]) == 0:
            pivots.append(node)
            pivot_nodes.add(node)
            pivot_nodes.update(seen_dictionary[node])




def random_init():
    global pivots, seen_dictionary
    individual = []
    for i in np.random.permutation(len(pivots)):
        pivot_watchers = list(seen_dictionary[pivots[i]])
        random_watcher = pivot_watchers[random.randrange(0, len(pivot_watchers))]
        individual.append((pivots[i], random_watcher))
    return individual


def create_model():
    global toolbox
    # model
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Permutation
    # toolbox.register("indices", random.sample, range(8), 8)
    toolbox.register("indices", random_init)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", eval)

    # register the crossover operator
    toolbox.register("mate", singlePointCrossoverTwoChilds)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", two_mutate_random)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    global toolbox

    create_model()
    # create an initial population of 100 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=200)
    generations = 50

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    # CXPB, MUTPB = 0.7, 0.1
    CXPB, MUTPB = 0.6, 0.15

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < generations:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i new individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        median = statistics.median(fits)

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        # print("  Std %s" % std)
        print("  Med %s" % median)
        with open('results.csv', mode='a') as results:
            results_writer = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow([g, min(fits), max(fits), mean, median])

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    return best_ind


def make_maze_from_file(map_file):
    lines = []
    for line in map_file:
        lines.append(line)
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    maze = [[0 for x in range(width)] for y in range(height)]
    for i in range(0, len(maze)):
        maze_row = lines[i+4]
        for j in range(0, len(maze[0])):
            cell = 1
            if maze_row[j] == ".":
                cell = 0
            maze[i][j] = cell
    # print_maze(maze)
    return maze

if __name__ == "__main__":
    map_file = open("maps/maze_11X11.map", "r")
    maze = make_maze_from_file(map_file)
    map_file.close()
    pre_processing()

    start_time = time.time()
    best_ind = main()
    print("--- %s seconds ---" % (time.time() - start_time))
    print_path(best_ind)
