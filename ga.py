import itertools
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from python_tsp.exact import solve_tsp_branch_and_bound
import random
import string
import time

random.seed(314159265)
CROSSOVER_PROBABILITY = 0.7
ELITE_INDIVIDUALS = 1
GENERATIONS = 550
GENOME_LENGTH = GRAPH_NODES = 45
MAX_EDGE_WEIGHT = 10
MIN_EDGE_WEIGHT = 1
MUTATION_RATE = 0.7 / GRAPH_NODES
POPULATION_SIZE = 150


alphabet = [c for c in string.printable]
graph_alphabet = alphabet[:GRAPH_NODES]
start_node = random.choice(graph_alphabet)


def compute_distance_matrix(weights):
    ordered_graph_alphabet = [start_node] + \
        [g for g in graph_alphabet if g != start_node]
    distance_matrix = np.zeros((len(graph_alphabet), len(graph_alphabet)))
    for i, n1 in enumerate(ordered_graph_alphabet):
        for j, n2 in enumerate(ordered_graph_alphabet):
            if (n1, n2) in weights:
                distance_matrix[i, j] = weights[(n1, n2)]
            elif (n2, n1) in weights:
                distance_matrix[i, j] = weights[(n2, n1)]
    return distance_matrix


# EDGCHFBAE
# and         cross_point = 4
# EGCBFHDAE

# becomes

# EDGCBFHAE   'EDGC' (first 4 of 1st) + 'BFHAE' (order of letters from 2nd not in 'EDGC')
# and
# EGCHFBDAE   'EGC' (all in first 4 of lst) + 'HFB' (remaining letters of 1st after 'EDGC') + 'D' (in first 4 of 1st) + 'AE' (remaining letters of 1st after 'EDGCHFB')
def crossover_path(path1, path2):
    cross_point = random.randrange(1, len(path1)-1)
    path1_first = path1[:cross_point]
    path1_rest = path1[cross_point:]
    path2_remaining = list(path2[:])
    [path2_remaining.remove(c) for c in path1_first]
    path2_remaining = "".join(path2_remaining)

    child1 = path1_first + \
        "".join(
            [c for c in path2_remaining if c not in path1_first or c == start_node])

    child2 = ""
    for c in path2[:-1]:
        if c not in path2_remaining[:-1]:
            child2 += c
        else:
            child2 += path1_rest[0]
            path1_rest = path1_rest[1:]
    child2 += start_node

    return (child1, child2)


def distance_traveled(path):
    return sum([weights[(path[i], path[i+1])] for i in range(len(path)-1)])


def distance_traveled_fitness(path):
    return sum(weights.values()) - sum([weights[(path[i], path[i+1])] for i in range(len(path)-1)])


def draw_graph(path, gen):
    """Based on:
    https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    """
    G = nx.Graph()
    for (node1, node2), edge_weight in weights.items():
        G.add_edge(node1, node2, weight=edge_weight)
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]

    pos = nx.kamada_kawai_layout(G, weight='weight')

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=6, edge_color=[
                           "red" if n1 + n2 in path or n2 + n1 in path else (0.1, 0.4, 0.7, 0.1) for (n1, n2) in edges])

    plt.gcf().set_size_inches(20, 10)
    ax = plt.gca()
    ax.margins(0.008)
    plt.axis("off")
    path_title = "".join(
        [path[0]] + [f"->{path[i]}" for i in range(1, len(path))])
    weights_title = "".join(["   " + str(weights[(path[0], path[1])])] + [
        " +" + str(weights[(path[i], path[i+1])]) for i in range(1, len(path)-1)])
    total_cost = distance_traveled(path)
    plt.title(f"Generation {gen}\n" + path_title + "\n" + weights_title +
              "\n" f"= {total_cost}", fontsize=16, fontweight='bold', fontfamily='monospace', color='blue')


def generate_random_individual():
    graph_alphabet_without_start_node = graph_alphabet[:]
    graph_alphabet_without_start_node.remove(start_node)
    return "".join([start_node] + random.sample(graph_alphabet_without_start_node, len(graph_alphabet) - 1) + [start_node])


def generate_random_weights(graph_alphabet):
    weights = {}

    for i, node in enumerate(graph_alphabet):
        for j, node2 in enumerate(graph_alphabet):
            if i != j and (graph_alphabet[j], graph_alphabet[i]) not in weights.keys():
                weights[(graph_alphabet[j], graph_alphabet[i])] = weights[(
                    graph_alphabet[i], graph_alphabet[j])] = random.randrange(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
    return weights


def graph_results(data, run):
    indices, mins, avgs, maxs, divs = zip(*data)
    fig, ax = plt.subplots()

    ax.plot(indices, mins, label='Min', linestyle='-')
    ax.plot(indices, avgs, label='Avg', linestyle='-')
    ax.plot(indices, maxs, label='Max', linestyle='-')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Min, Avg, and Max Fitness Over Generations')

    ax.legend()

    plt.gcf().set_size_inches(12, 8)
    plt.grid(True)
    # plt.savefig(f'plot{run}_not_mega_mutate.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(indices, divs, label='Diversity', linestyle='-')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity')
    ax.set_title('Diversity Over Generations')

    ax.legend()
    plt.grid(True)
    # plt.savefig(f'div{run}_not_mega_mutate.png')
    plt.show()


def hamming_distance(path1, path2):
    path1, path2 = list(path1), list(path2)
    return sum(p1 != p2 for p1, p2 in zip(path1, path2))


def mega_mutate(pop):
    return [multiple_mutate(p, 10) if random.random() < 0.3 else multiple_mutate(p, 2) for p in pop]


def multiple_mutate(path, n):
    for i in range(n):
        path = mutate_path_swap_two(path)
    return path


def mutate_path_swap_two(path):
    new_path = [c for c in path]
    for i in range(1, len(path)-1):
        if random.random() < MUTATION_RATE:
            j = random.randrange(1, len(path)-1)
            while i == j:
                j = random.randrange(1, len(path)-1)
            temp = new_path[i]
            new_path[i] = new_path[j]
            new_path[j] = temp
    return "".join(new_path)


def pmx_crossover(path1, path2):
    """
    Based on: https://user.ceng.metu.edu.tr/~ucoluk/research/publications/tspnew.pdf
    """
    path1, path2 = path1[1:-1], path2[1:-1]
    cross_point = random.randrange(1, len(path1)-1)
    child1 = list(path1[:])
    for i in range(cross_point):
        temp = child1[i]
        temp_idx = child1.index(path2[i])
        child1[i] = path2[i]
        child1[temp_idx] = temp

    child2 = list(path2[:])
    for i in range(cross_point):
        temp = child2[i]
        temp_idx = child2.index(path1[i])
        child2[i] = path1[i]
        child2[temp_idx] = temp
    return (start_node + "".join(child1) + start_node,  start_node + "".join(child2) + start_node)


def population_diversity(population):
    total_distance = 0
    num_pairs = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            total_distance += hamming_distance(population[i], population[j])
            num_pairs += 1

    diversity = total_distance / num_pairs

    return diversity


def ranked_select(pop, single=False):
    pop = pop[::-1]
    fitness_weights = [i / (len(pop)*(len(pop)+1)/2) for i in range(len(pop))]
    if single:
        return random.choices(pop, weights=fitness_weights)[0]

    return random.choices(pop, weights=fitness_weights, k=len(pop))


def roulette_wheel_select(pop, single=False):
    pop_total_fitness = sum([1 / distance_traveled(c) for c in pop])
    fitness_weights = [(1 / distance_traveled(c)) /
                       pop_total_fitness for c in pop]
    if single:
        return random.choices(pop, weights=fitness_weights)[0]

    return random.choices(pop, weights=fitness_weights, k=len(pop))


def run_simulation():
    global MUTATION_RATE
    global fittest_individuals_each_generation

    for i in range(runs):
        fittest_individuals_each_generation = []
        crossover = True
        global start_node
        global weights
        start_node = random.choice(graph_alphabet)
        weights = generate_random_weights(graph_alphabet)
        dist_matrix = compute_distance_matrix(weights)

        start = time.time()
        min_path = (0, 0)
        if GENOME_LENGTH <= 20:
            optimal_path, optimal_cost = solve_tsp_branch_and_bound(
                dist_matrix)

            ordered_alphabet = [start_node] + \
                [g for g in graph_alphabet if g != start_node]
            optimal_path = "".join([ordered_alphabet[int(i)]
                                   for i in optimal_path]) + start_node

            min_path = (optimal_path, optimal_cost)

        end = time.time()

        print(f"Brute force TSP took {end-start} seconds")

        population = [generate_random_individual()
                      for j in range(POPULATION_SIZE)]

        first_solution = -1

        data = []
        for generation in range(GENERATIONS):
            fitnesses = [fitness_func(x) for x in population]
            max_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(population)
            min_fitness = min(fitnesses)
            fitness_diversity = population_diversity(population)
            data.append((generation, min_fitness, avg_fitness,
                        max_fitness, fitness_diversity))

            if generation % int(0.01*GENERATIONS) == 0 and fitness_diversity < 0.2 * GENOME_LENGTH:
                population = mega_mutate(population)

            if sum(weights.values()) - max_fitness == int(min_path[1]) and first_solution == -1:
                first_solution = generation
                
            print(
                f"Shortest path (fittest) of generation {generation} is {sum(weights.values()) - max_fitness} (best_possible={min_path[1]}, best_possible_path={min_path[0]})")
            
            population.sort(key=lambda x: fitness_func(x), reverse=True)
            fittest_individuals_each_generation.append(population[0])
            elite = population[:ELITE_INDIVIDUALS]
            if crossover:
                next_generation = []
                for crossover_chance in range(int(len(population) / 2)):
                    parent1 = parent_selection_func(
                        population, single=True)
                    parent2 = parent_selection_func(
                        population, single=True)
                    while parent1 == parent2:
                        parent2 = parent_selection_func(
                            population, single=True)
                    if random.random() < CROSSOVER_PROBABILITY:
                        offspring = crossover_path(parent1, parent2)
                        next_generation.extend(offspring)
                    else:
                        next_generation.extend((parent1, parent2))

                population = next_generation

            population = elite + \
                [mutate_path_swap_two(c) if random.random(
                ) < MUTATION_RATE else c for c in population[ELITE_INDIVIDUALS:]]

        end = time.time()

        print(f"Run took {end-start} seconds")
        population.sort(key=lambda x: fitness_func(x), reverse=True)

        print("\nTop five:")
        [print(p, distance_traveled(p)) for p in population[:5]]

        print(f"Generation with first sol: {first_solution}")

        # [print(f"{len(p)} of {p[0]}, fitness = {distance_traveled(''.join(p[0]))}") for p in [
        #     [j for i in range(population.count(j))] for j in list(set(population))]]

        graph_results(data, i)
        print(f"Fittest ever: {min(fittest_individuals_each_generation)}")
        # Animate the path over the generations
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=len(
            fittest_individuals_each_generation), repeat=False, interval=20, cache_frame_data=False)
        # # animation_file = 'animation.gif'
        # # ani.save(animation_file, writer='pillow')
        plt.show()


def truncation_select(pop, single=False):
    if single:
        return random.choice(pop)
    return pop[:int(len(pop)/2)] + pop[:int(len(pop)/2)]


def tsp_bruteforce(nodes, weights):
    nodes = nodes[1:-1]
    all_permutations = itertools.permutations(nodes)
    min_distance = float('inf')
    best_path = None

    all_permutations = ["".join(p) for p in all_permutations]
    all_permutations = [start_node + perm +
                        start_node for perm in all_permutations]

    for permutation in all_permutations:
        total_distance = distance_traveled(permutation)
        if total_distance < min_distance:
            min_distance = total_distance
            best_path = permutation

    return best_path, min_distance


def update(frame):
    plt.clf()
    path = fittest_individuals_each_generation[frame]
    draw_graph(path, frame)


fitness_func = distance_traveled_fitness
fittest_individuals_each_generation = []
parent_selection_func = ranked_select
runs = 10
weights = generate_random_weights(graph_alphabet)


run_simulation()
