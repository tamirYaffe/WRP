import random, copy, statistics, time, csv
from tkinter import *
from deap import base
from deap import creator
from deap import tools
import networkx as nx
from bresenham import bresenham

# define the maze.
from GUI import CellGrid
toolbox = base.Toolbox()
G = nx.Graph()
all_seen_size = 0
all_shortest_paths = []
seen_dictionary = {}
seen_all = False


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
        seen = []
        for i in range(0, len(maze)):
            for j in range(0, len(maze[0])):
                if maze[i][j] == 1:
                    continue
                other = Position(j, i)
                if self.is_exsits_los(other) or other.is_exsits_los(self):
                    seen.append(other)
        return seen

    def __eq__(self, other):
        return self.x is other.x and self.y is other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str(self.x) + "," + str(self.y)

maze = [[0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]]

# define start position
entry_position = Position(6, 7)

def three_mutate_random(individual):
    num = random.random()
    if num < 0.33:
        onePointMutate(individual)
    elif num < 0.66 and not seen_all:
        addMoveMutate(individual)
    else:
        removeMoveMutate(individual)


def onePointMutate(individual):
    index = random.randint(0, all_seen_size - 1)
    move = random.randint(0, 3)
    while individual[index] is move:
        move = random.randint(0, 3)
    individual[index] = move


def find_max_move(current_position, min_pivot):
    max_move = -1
    min_dist = 10000
    for move in range(0,4):
        current_position = perform_move(current_position, move)
        if current_position.x < 0 or current_position.x > len(maze[0]) - 1 or current_position.y < 0 \
                or current_position.y > len(maze) - 1 or maze[current_position.y][current_position.x] is 1:
            continue
        distance = all_shortest_paths[current_position][min_pivot]
        if distance < min_dist:
            min_dist = distance
            max_move = move
    return max_move


def addMoveMutate(individual):
    seen_set = set()
    seen_set.update(seen_dictionary[entry_position])
    current_position = Position(entry_position.x, entry_position.y)
    prev_position = Position(current_position.x, current_position.y)
    path_positions = [copy.deepcopy(current_position)]
    min_dist_to_pivots = {}
    min_dist_index_to_pivots = {}
    for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
        min_dist_to_pivots[pivot] = 100000
        min_dist_index_to_pivots[pivot] = -1
    index = -1
    for move in individual:
        index = index + 1
        current_position = perform_move(current_position, move)
        if current_position.x < 0 or current_position.x > len(maze[0]) - 1 or current_position.y < 0 \
                or current_position.y > len(maze) - 1 or maze[current_position.y][current_position.x] == 1:
            current_position = copy.deepcopy(prev_position)
        else:
            seen_set.update(seen_dictionary[current_position])
            for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
                if pivot not in seen_set:
                    min_vis_dist_from_pivot(min_dist_to_pivots, min_dist_index_to_pivots, pivot, current_position,
                                            index)
        prev_position = copy.deepcopy(current_position)
        path_positions.append(copy.deepcopy(current_position))

    min_dist_pivot = 10000
    min_dist_index = -1
    min_pivot = None
    for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
        if pivot not in seen_set:
            if min_dist_to_pivots[pivot] < min_dist_pivot:
                min_dist_pivot = min_dist_to_pivots[pivot]
                min_dist_index = min_dist_index_to_pivots[pivot]
                min_pivot = pivot
    if min_pivot is None:
        return
    index = min_dist_index
    current_position = path_positions[index]
    move = find_max_move(current_position, min_pivot)
    # index = random.randint(0, all_seen_size - 1)
    # move = random.randint(0, 3)
    while index < len(individual):
        temp_move = individual[index]
        individual[index] = move
        move = temp_move
        index = index + 1


def removeMoveMutate(individual):
    index = random.randint(0, all_seen_size - 1)
    last_move = random.randint(0, 3)
    while index + 1 < len(individual):
        individual[index] = individual[index + 1]
        index = index + 1
    individual[all_seen_size - 1] = last_move





# the goal ('fitness') function to be maximized
# 0-right, 1- left, 2- up, 3- down.
def perform_move(current_position, move):
    if move is 0:
        current_position.x = current_position.x + 1
    if move is 1:
        current_position.x = current_position.x - 1
    if move is 2:
        current_position.y = current_position.y - 1
    if move is 3:
        current_position.y = current_position.y + 1
    return current_position


def min_vis_dist_from_pivot(min_dist_to_pivots, min_dist_position_to_pivots, pivot, current_position, index):
    for position in seen_dictionary[pivot]:
        distance = all_shortest_paths[current_position][position]
        if distance < min_dist_to_pivots[pivot]:
            min_dist_to_pivots[pivot] = distance
            min_dist_position_to_pivots[pivot] = index


def eval(individual):
    global seen_all
    seen_set = set()
    seen_set.update(seen_dictionary[entry_position])
    penalty = 0
    bonus = 0
    current_position = Position(entry_position.x, entry_position.y)
    prev_position = Position(current_position.x, current_position.y)
    move_ctr = 0
    path_positions = []
    min_dist_to_pivots = {}
    min_dist_index_to_pivots = {}
    for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
        min_dist_to_pivots[pivot] = 100000
        min_dist_index_to_pivots[pivot] = -1
    index = -1
    for move in individual:
        index = index + 1
        current_position = perform_move(current_position, move)
        # check for penalty
        if current_position.x < 0 or current_position.x > len(maze[0]) - 1 or current_position.y < 0 \
                or current_position.y > len(maze) - 1 or maze[current_position.y][current_position.x] is 1:
            penalty = penalty + 100
            current_position = copy.deepcopy(prev_position)
        else:
            seen_set.update(seen_dictionary[current_position])
            for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
                if pivot not in seen_set:
                    min_vis_dist_from_pivot(min_dist_to_pivots, min_dist_index_to_pivots, pivot, current_position,
                                            index)
        prev_position = copy.deepcopy(current_position)
        path_positions.append(copy.deepcopy(current_position))
        move_ctr = move_ctr + 1

        if len(seen_set) == all_seen_size:
            seen_all = True
            clean_individual(individual, path_positions)
            bonus = -10000
            break
    unseen_pivots = 0
    for pivot in [Position(7, 1), Position(0, 6), Position(0, 4), Position(1, 0)]:
        if pivot not in seen_set:
            unseen_pivots = unseen_pivots + 1
            penalty = penalty + 10000 * (min_dist_to_pivots[pivot]) + 1000 * (
                        all_seen_size - min_dist_index_to_pivots[pivot])
    # if unseen_pivots > 2:
    #     penalty = penalty / unseen_pivots
    return (all_seen_size - len(seen_set)) * 1000 + move_ctr * 10 + penalty + bonus,


def clean_individual(individual, path_positions):
    i = 0
    prev_position = Position(entry_position.x, entry_position.y)
    for position in path_positions:
        move = (position.x - prev_position.x, position.y - prev_position.y)
        prev_position = position
        move_num = -1
        if move == (0, 0):
            continue
        if move == (1, 0):
            move_num = 0
        if move == (-1, 0):
            move_num = 1
        if move == (0, -1):
            move_num = 2
        if move == (0, 1):
            move_num = 3
        individual[i] = move_num
        i = i + 1


def print_path(best_ind):
    current_position = Position(entry_position.x, entry_position.y)
    maze[current_position.y][current_position.x] = 2
    seen_set = set()
    seen_set.update(seen_dictionary[entry_position])
    prev_position = Position(current_position.x, current_position.y)
    for move in best_ind:
        current_position = perform_move(current_position, move)
        # check for penalty
        if current_position.x < 0 or current_position.x > len(maze[0]) - 1 or current_position.y < 0 \
                or current_position.y > len(maze) - 1 or maze[current_position.y][current_position.x] is 1:
            current_position = copy.deepcopy(prev_position)
            continue
        else:
            prev_position = copy.deepcopy(current_position)
            maze[current_position.y][current_position.x] = 2
            seen_set.update(seen_dictionary[current_position])
        if len(seen_set) == all_seen_size:
            break
    # for row in maze:
    #     print(row)
    print("--- %s coverage ---" % (len(seen_set) / all_seen_size))
    for cell in seen_dictionary:
        if cell not in seen_set:
            print(cell)
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
    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    comp = nx.connected_components(G)
    print(comp)


def create_model():
    global toolbox
    # model
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # List
    # 0-right, 1- left, 2- up, 3- down.
    toolbox.register("attr_int", random.randint, 0, 3)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_int, 42)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", eval)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mate", singlePointCrossover)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", three_mutate_random)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    global toolbox

    pre_processing()
    create_model()
    random.seed(128)
    # create an initial population of 100 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=1000)
    generations = 75

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


if __name__ == "__main__":
    start_time = time.time()
    best_ind = main()
    print("--- %s seconds ---" % (time.time() - start_time))
    print_path(best_ind)
