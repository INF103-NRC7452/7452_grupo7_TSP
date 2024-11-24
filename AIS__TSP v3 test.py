import os
import csv
import random
import time
import statistics
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# AIS (Artificial Immune System) configurations
POPULATION_SIZE = 50
CLONES_PER_ANTIBODY = 10
MUTATION_RATE = 0.01
GENERATIONS = 5000
NUM_EXECUTIONS = 31

class AIS:
    def __init__(self, distance_matrix, is_symmetric=True, population_size=POPULATION_SIZE, clones_per_antibody=CLONES_PER_ANTIBODY, mutation_rate=MUTATION_RATE, generations=GENERATIONS, seed=None):
        self.distance_matrix = distance_matrix
        self.is_symmetric = is_symmetric
        self.n_cities = len(distance_matrix)
        self.population_size = population_size
        self.clones_per_antibody = clones_per_antibody
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize_population(self):
        population = np.array([np.random.permutation(self.n_cities) for _ in range(self.population_size)])
        return population

    def clone_and_mutate(self, antibody):
        clones = np.array([np.copy(antibody) for _ in range(self.clones_per_antibody)])
        for clone in clones:
            if random.random() < self.mutation_rate:
                # Mutation by swapping two cities
                i, j = np.random.choice(self.n_cities, size=2, replace=False)
                clone[i], clone[j] = clone[j], clone[i]
        return clones

    def evaluate(self, antibody):
        total_distance = 0
        for i in range(len(antibody) - 1):
            city_a = antibody[i]
            city_b = antibody[i + 1]
            total_distance += self.distance_matrix[city_a][city_b]
        # Return to the starting city if symmetric TSP
        if self.is_symmetric:
            total_distance += self.distance_matrix[antibody[-1]][antibody[0]]
        return total_distance  # Minimization problem

    def select_best_antibodies(self, antibodies, num_best=50):
        fitness_values = np.array([self.evaluate(antibody) for antibody in antibodies])
        best_indices = np.argsort(fitness_values)[:num_best]
        return antibodies[best_indices]

def read_tsp_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')

    node_coord_section = False
    edge_weight_section = False
    edge_data_section = False
    node_coord_type = None
    edge_weight_format = None
    edge_weight_type = None
    dimension = None
    node_coords = {}
    edge_weights = []
    problem_type = 'TSP'
    is_symmetric = True
    instance_name = None

    # Parse specification part
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith('EOF'):
            # End of the file
            break

        parts = line.split(':')
        if len(parts) == 2:
            keyword = parts[0].strip()
            value = parts[1].strip()

            if keyword == 'NAME':
                instance_name = value
            elif keyword == 'TYPE':
                problem_type = value
                if problem_type == 'ATSP':
                    is_symmetric = False
            elif keyword == 'DIMENSION':
                dimension = int(value)
            elif keyword == 'EDGE_WEIGHT_TYPE':
                edge_weight_type = value
            elif keyword == 'EDGE_WEIGHT_FORMAT':
                edge_weight_format = value
            elif keyword == 'NODE_COORD_TYPE':
                node_coord_type = value
            else:
                # Not needed for the solution or unknown keyword
                pass

        if line.startswith('NODE_COORD_SECTION'):
            node_coord_section = True
            i += 1
            break
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            edge_weight_section = True
            i += 1
            break
        elif line.startswith('EDGE_DATA_SECTION'):
            edge_data_section = True
            i += 1
            break
        i += 1

    # Parse data part
    while i < len(lines) and (node_coord_section or edge_weight_section or edge_data_section):
        line = lines[i].strip()

        if not line:
            i += 1
            continue
        if line.startswith('EOF'):
            break
        if node_coord_section:
            if line.startswith('DEMAND_SECTION') or line.startswith('EDGE_WEIGHT_SECTION') or line.startswith('EOF'):
                # End of node coord section
                node_coord_section = False
                # Possibly edge weight section starts
                if line.startswith('EDGE_WEIGHT_SECTION'):
                    edge_weight_section = True
                i += 1
                continue
            parts = line.split()
            if len(parts) == 3:
                index = int(parts[0]) - 1  # adjust index for 0-based
                x = float(parts[1])
                y = float(parts[2])
                node_coords[index] = (x, y)
            if len(node_coords) == dimension:
                node_coord_section = False
        elif edge_weight_section:
            if line.startswith('DISPLAY_DATA_SECTION') or line.startswith('EOF'):
                # End of edge weight section
                edge_weight_section = False
                i += 1
                continue
            # Accumulate edge weights
            edge_weights_line = line.split()
            if edge_weights_line:
                edge_weights.extend(map(float, edge_weights_line))
            if len(edge_weights) >= dimension * dimension:
                edge_weight_section = False
        elif edge_data_section:
            if line.startswith('EOF'):
                # End of edge data section
                edge_data_section = False
                break
        i += 1

    if dimension is None:
        raise ValueError("Dimension not specified in TSPLIB file.")

    # If no edge_weight_type is specified, default to EUC_2D (common case)
    if not edge_weight_type:
        if node_coord_type in ['TWOD_COORDS', 'THREED_COORDS']:
            edge_weight_type = 'EUC_2D' if node_coord_type == 'TWOD_COORDS' else 'EUC_3D'
        else:
            edge_weight_type = 'EXPLICIT'

    # Construct the distance matrix
    if edge_weight_type == 'EXPLICIT' and edge_weight_format and edge_weights:
        distance_matrix = parse_explicit_edge_weights(edge_weights, dimension, edge_weight_format)
    elif edge_weight_type in ['EUC_2D', 'CEIL_2D', 'GEO', 'ATT']:
        coords = [node_coords[i] for i in range(dimension)]
        distance_matrix = compute_distance_matrix(coords, edge_weight_type)
    else:
        raise ValueError(f"Unsupported or incomplete handling for EDGE_WEIGHT_TYPE: {edge_weight_type}")

    return instance_name, distance_matrix, is_symmetric

def parse_explicit_edge_weights(edge_weights, dimension, edge_weight_format):
    distance_matrix = np.zeros((dimension, dimension))
    idx = 0

    if edge_weight_format == 'FULL_MATRIX':
        # Weights are given by a full matrix
        if len(edge_weights) != dimension * dimension:
            raise ValueError("Explicit edge weights do not match dimension for FULL_MATRIX format.")
        for i in range(dimension):
            for j in range(dimension):
                distance_matrix[i][j] = edge_weights[idx]
                idx += 1
    elif edge_weight_format == 'UPPER_ROW':
        # Upper triangular matrix (row-wise without diagonal entries)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'LOWER_ROW':
        # Lower triangular matrix (row-wise without diagonal entries)
        for i in range(dimension):
            for j in range(i):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    return distance_matrix

def compute_distance_matrix(coords, edge_weight_type):
    n = len(coords)
    distance_matrix = np.zeros((n, n))

    if edge_weight_type == 'EUC_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'CEIL_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                distance = math.ceil(dist)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'GEO':
        coords_rad = [convert_geo_coords(coord) for coord in coords]
        for i in range(n):
            lati, loni = coords_rad[i]
            for j in range(i + 1, n):
                latj, lonj = coords_rad[j]
                distance = geo_distance(lati, loni, latj, lonj)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'ATT':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                xd = xi - xj
                yd = yi - yj
                rij = math.sqrt(((xd * xd) + (yd * yd)) / 10.0)
                tij = round(rij)
                distance = tij + 1 if tij < rij else tij
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}")

    return distance_matrix

def convert_geo_coords(coord):
    PI = 3.141592653589793
    deg = int(coord[0])
    minutes = coord[0] - deg
    latitude = PI * (deg + 5.0 * minutes / 3.0) / 180.0
    deg = int(coord[1])
    minutes = coord[1] - deg
    longitude = PI * (deg + 5.0 * minutes / 3.0) / 180.0
    return latitude, longitude

def geo_distance(lati, loni, latj, lonj):
    RRR = 6378.388
    q1 = math.cos(loni - lonj)
    q2 = math.cos(lati - latj)
    q3 = math.cos(lati + latj)
    dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    return dij

def ais_search(distance_matrix, is_symmetric, population_size, clones_per_antibody, mutation_rate, generations, seed, run_num, instance_name):
    # Initialize AIS with the provided parameters
    ais_system = AIS(
        distance_matrix=distance_matrix,
        is_symmetric=is_symmetric,
        population_size=population_size,
        clones_per_antibody=clones_per_antibody,
        mutation_rate=mutation_rate,
        generations=generations,
        seed=seed
    )
    population = ais_system.initialize_population()

    best_solution = None
    best_fitness = float('inf')  # Minimization problem; lower distance is better

    iteration_results = []
    start_time = time.time()

    # Setup CSV file for iteration results
    results_file_path = f'results/results_{instance_name}_{run_num}.csv'
    os.makedirs('results', exist_ok=True)
    with open(results_file_path, 'w', newline='') as csvfile:
        fieldnames = ['instance_name', 'run_num', 'iteration', 'seed', 'fitness', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for iteration in range(1, generations + 1):
        iteration_start_time = time.time()

        # Clone and mutate all antibodies
        cloned_population = []
        for antibody in population:
            clones = ais_system.clone_and_mutate(antibody)
            cloned_population.append(clones)
        cloned_population = np.vstack(cloned_population)

        # Select best antibodies for the next generation
        population = ais_system.select_best_antibodies(cloned_population, num_best=population_size)

        # Evaluate best individual in the current population
        current_best_fitness = float('inf')
        for antibody in population:
            dist = ais_system.evaluate(antibody)
            if dist < current_best_fitness:
                current_best_fitness = dist

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[0]

        iteration_time = time.time() - iteration_start_time
        cumulative_time = time.time() - start_time

        # Record iteration data
        iteration_results.append({
            'instance_name': instance_name,
            'run_num': run_num,
            'iteration': iteration,
            'seed': seed,
            'fitness': best_fitness,
            'time': cumulative_time
        })

        # Write iteration result to CSV
        with open(results_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['instance_name', 'run_num', 'iteration', 'seed', 'fitness', 'time'])
            writer.writerow(iteration_results[-1])

    return best_solution, best_fitness, iteration_results

def run_ais_for_instance(distance_matrix, is_symmetric, instance_name, population_size, clones_per_antibody, mutation_rate, generations, num_executions):
    # Setup directories
    os.makedirs('results', exist_ok=True)

    final_statistics = []

    # Parallel execution for multiple runs
    with ProcessPoolExecutor(max_workers=num_executions) as executor:
        seeds = [random.randint(1, 1000000) for _ in range(num_executions)]
        futures = []

        for run_num, seed in enumerate(seeds, start=1):
            futures.append(
                executor.submit(
                    ais_search,
                    distance_matrix,
                    is_symmetric,
                    population_size,
                    clones_per_antibody,
                    mutation_rate,
                    generations,
                    seed,
                    run_num,
                    instance_name
                )
            )

        results = [f.result() for f in futures]

    # Process results from all runs
    for (run_num, seed), (best_solution, best_fitness, iteration_results) in zip(enumerate(seeds, start=1), results):
        # The iteration results are already saved in CSV per iteration

        # Compute per-run summary statistics using iteration results (like TS approach)
        fitness_values = [data['fitness'] for data in iteration_results]  # Fitness over all iterations for this run
        time_values = [data['time'] for data in iteration_results]       # Times over all iterations for this run

        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        avg_fitness = statistics.mean(fitness_values) if fitness_values else float('inf')
        fitness_std_dev = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
        avg_time = statistics.mean(time_values) if time_values else 0

        final_statistics.append({
            'run_num': run_num,
            'final_fitness': best_fitness,
            'min_fitness': min_fitness,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'fitness_std_dev': fitness_std_dev,
            'avg_time': avg_time,
            'seed': seed
        })

    return final_statistics

def save_ais_final_statistics(instance_name, final_statistics):
    # Save final statistics to CSV
    statistics_file = f'results/statistics_{instance_name}.csv'
    with open(statistics_file, 'w', newline='') as csvfile:
        fieldnames = [
            'instance_name', 'run_num', 'final_fitness', 'min_fitness', 
            'max_fitness', 'avg_fitness', 'fitness_std_dev', 'avg_time'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in final_statistics:
            writer.writerow({
                'instance_name': instance_name,
                'run_num': stat['run_num'],
                'final_fitness': stat['final_fitness'],
                'min_fitness': stat['min_fitness'],
                'max_fitness': stat['max_fitness'],
                'avg_fitness': stat['avg_fitness'],
                'fitness_std_dev': stat['fitness_std_dev'],
                'avg_time': stat['avg_time']
            })

def solve_tsp_files_with_ais(folder_path, file_names, num_executions, generations, population_size, clones_per_antibody, mutation_rate):
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name + ".tsp")
        print(f"Solving TSP for {file_name} with AIS...\n")

        # Read the TSP instance from the TSPLIB file
        try:
            instance_name, distance_matrix, is_symmetric = read_tsp_instance(file_path)
            if not is_symmetric:
                print(f"Instance {instance_name} is not symmetric. This code only supports symmetric TSP at the moment.")
                continue

            # Run the AIS
            final_statistics = run_ais_for_instance(
                distance_matrix,
                is_symmetric,
                instance_name,
                population_size,
                clones_per_antibody,
                mutation_rate,
                generations,
                num_executions
            )

            # Save final statistics for this instance
            save_ais_final_statistics(instance_name, final_statistics)

            # Print summary of the best found solution across runs
            best_solution_overall = min(final_statistics, key=lambda x: x['final_fitness'])
            print(f"Instance {instance_name} solved with AIS. Best overall fitness: {best_solution_overall['final_fitness']:.2f}\n")

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

if __name__ == '__main__':
    # Example usage
    folder_path = 'tsplib'
    file_names = ["ulysses16", "ulysses22", "gr202", "tsp225", "a280", "pcb442", "gr666"]
    num_executions = NUM_EXECUTIONS
    generations = GENERATIONS
    population_size = POPULATION_SIZE
    clones_per_antibody = CLONES_PER_ANTIBODY
    mutation_rate = MUTATION_RATE

    solve_tsp_files_with_ais(folder_path, file_names, num_executions, generations, population_size, clones_per_antibody, mutation_rate)
