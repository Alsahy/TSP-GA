import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 500
MUTATION_RATE = 0.25
IND_LENGTH = 12
GENERATIONS = 250
HEIGHT = 750
WIDTH = 1250
MINIMUM_DISTANCE = 1e9
MINIMUM_INDEX = 0

"""
The GA Receipt:
1- Generate the cities themself which doesn't represent a population or a solution
2- Generate the random order of traversing the cities
3- Generate population from random individuals each one is a different order
4- Natural select from the population depending on fitness
5- Perform crossover to get the new generation
6- Perform mutation to explore more solutions 
7- Clone the new generation into the population
"""

# For City Number : (random x, random y)
cities = {i : (random.randint(0, WIDTH), random.randint(0, HEIGHT)) for i in range(IND_LENGTH)}
fitness = [0 for _ in range(POPULATION_SIZE)]
distances = [[0] * POPULATION_SIZE for _ in range(POPULATION_SIZE)]
growth_indicator = []

def calculate_distances():
    def get_distance(a, b):
        x_diff = abs(a[0] - b[0])
        y_diff = abs(a[1] - b[1])
        return (x_diff ** 2 + y_diff ** 2) ** 0.5

    for i in range(IND_LENGTH):
        for j in range(IND_LENGTH):
            distances[i][j] = get_distance(cities[i], cities[j])

calculate_distances()


def generate_ind():
    ind = [i for i in range(IND_LENGTH)]
    random.shuffle(ind)
    return ind

def generate_population():
    return [generate_ind() for i in range(POPULATION_SIZE)]
def update_fitness(population):
    global fitness
    def calculate_fitness(ind):
        global distances
        total = 0
        for i in range(0, IND_LENGTH - 1):
            total += distances[ind[i]][ind[i+1]]
        return 1 / (total ** 2)

    def normalize_fitness():
        global fitness
        s = 0
        for i in range(POPULATION_SIZE):
            s += fitness[i]
        fitness = [i/s for i in fitness]

    for i in range(POPULATION_SIZE):
        fitness[i] = calculate_fitness(population[i])
    normalize_fitness()


def natural_selction(population):
    update_fitness(population)
    selected_population = []
    for i in range(POPULATION_SIZE):
        accumulator = 0
        random_pick = random.random()
        for j in range(POPULATION_SIZE):
            accumulator += fitness[j]
            if random_pick <= accumulator:
                selected_population.append(population[j])
                break
    return selected_population

def children(parents):
    def crossover(p1, p2):
        rbegin = random.randint(0, IND_LENGTH-1)
        rend = random.randint(rbegin, IND_LENGTH)
        child1 = p1[rbegin:rend]
        for i in p2:
            if i not in child1:
                child1.append(i)
        rbegin = random.randint(0, IND_LENGTH - 1)
        rend = random.randint(rbegin, IND_LENGTH)
        child2 = p2[rbegin:rend]
        for i in p1:
            if i not in child2:
                child2.append(i)
        return child1, child2

    children = []
    for i in range(0, POPULATION_SIZE, 2):
        p1, p2 = parents[i], parents[i+1]
        c1, c2 = crossover(p1, p2)
        children.append(c1)
        children.append(c2)

    return children

def mutation(population):
    def mutate(ind):
        rand_indx = random.randint(0, IND_LENGTH-1)
        ind[rand_indx], ind[(rand_indx+1)%IND_LENGTH] = ind[(rand_indx+1)%IND_LENGTH], ind[rand_indx]
        return ind

    for i in range(POPULATION_SIZE):
        p = random.random()
        if p <= MUTATION_RATE:
            mutate(population[i])

    return population

def calc_distance(population):
    global MINIMUM_DISTANCE
    for i in range(POPULATION_SIZE):
        total = 0
        for j in range(0, IND_LENGTH-1):
            total += distances[population[i][j]][population[i][j+1]]
        if total < MINIMUM_DISTANCE:
            MINIMUM_DISTANCE = total
            MINIMUM_INDEX = i

def draw(path):
    global cities
    img = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)
    for i in cities.keys():
        cv2.circle(img, cities[i], 10, (0, 0, 200), -1)

    for indx in range(IND_LENGTH-1):
        cv2.line(img, cities[path[indx]], cities[path[indx+1]], (255, 255, 255), 3)

    cv2.imshow("Traveling Salesman Problem", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def genetic_algorithms(times):
    population = generate_population()
    for t in range(times):
        selected_population = natural_selction(population)
        new_generation = children(selected_population)
        mutated_population = mutation(new_generation)
        population = mutated_population[:POPULATION_SIZE//2] + selected_population[POPULATION_SIZE//2:]
        calc_distance(population)
        growth_indicator.append(MINIMUM_DISTANCE)
        print("BEST DISTANCE IN : " + str(t+1) + " = " + str(MINIMUM_DISTANCE))

    draw(population[MINIMUM_INDEX])
    plt.plot(range(1, GENERATIONS + 1), growth_indicator, marker='x')
    plt.title('Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Distance')
    plt.show()

genetic_algorithms(GENERATIONS)
