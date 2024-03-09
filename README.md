**Traveling Salesman Problem with Genetic Algorithm**

This code implements a Genetic Algorithm (GA) to solve the Traveling Salesman Problem (TSP). The TSP aims to find the shortest possible route that visits each city exactly once and returns to the starting city.

**Features:**

- Generates random cities with random coordinates.
- Creates a population of individuals, where each individual represents a possible route order for visiting the cities.
- Employs natural selection to choose fitter individuals (shorter routes) for reproduction.
- Uses cycle crossover to create offspring by exchanging segments between parent routes.
- Introduces mutation to maintain diversity in the population.
- Tracks the best distance found in each generation.
- Visualizes the final best route and the overall improvement across generations.

**How to Run:**

1. Save the code as a Python file (e.g., tsp_ga.py).
2. Run the script from the command line: `python tsp_ga.py`

**Code Structure:**

- **Configuration:**
    - `POPULATION_SIZE`: Number of individuals in the population.
    - `MUTATION_RATE`: Probability of an individual undergoing mutation.
    - `IND_LENGTH`: Number of cities (length of the route).
    - `GENERATIONS`: Number of generations to evolve the population.
    - `HEIGHT` and `WIDTH`: Dimensions of the area where cities are located.
- **City Generation:**
    - `cities`: Dictionary containing city indices and their corresponding (x, y) coordinates.
- **Distance Calculation:**
    - `calculate_distances`: Pre-computes the distance between all pairs of cities.
    - `get_distance`: Calculates the Euclidean distance between two cities.
- **Individual and Population Generation:**
    - `generate_ind`: Creates an individual by shuffling a list of city indices (representing the visiting order).
    - `generate_population`: Generates a population of individuals.
- **Fitness Evaluation:**
    - `calculate_fitness`: Calculates the fitness of an individual (reciprocal of the total route distance squared).
    - `update_fitness`: Updates the fitness of all individuals in the population and normalizes their fitness values.
- **Selection:**
    - `natural_selection`: Implements roulette wheel selection based on individual fitness.
- **Crossover:**
    - `crossover`: Performs cycle crossover on two parent individuals to create offspring.
    - `children`: Generates offspring for the entire population using crossover.
- **Mutation:**
    - `mutation`: Applies random swap mutation to an individual with a certain probability.
- **Distance Tracking:**
    - `calc_distance`: Finds the individual with the shortest route distance in the current population.
    - `MINIMUM_DISTANCE` and `MINIMUM_INDEX`: Global variables to track the best distance and its corresponding individual index.
- **Visualization:**
    - `draw`: Visualizes the best route found using OpenCV and matplotlib.
- **Genetic Algorithm Main Loop:**
    - `genetic_algorithms`: Runs the GA for a specified number of generations.
      - Initializes the population.
      - Iterates through generations:
        - Performs selection to choose fitter individuals.
        - Creates offspring using crossover.
        - Introduces mutation.
        - Updates the population with the next generation.
        - Tracks the best distance found so far.
      - Visualizes the final best route and the improvement over generations.

**Note:**

- This is a basic implementation of a GA for TSP. You can experiment with different parameters and crossover/mutation techniques to potentially improve performance.
