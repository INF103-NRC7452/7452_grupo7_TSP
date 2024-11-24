import os
import csv
import glob
import random
import time
import statistics
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Genetic Algorithm configurations
POPULATION_SIZE = 300
ELITE_SIZE = 0.15
MUTATION_RATE = 0.01
GENERATIONS = 5000
NUM_EXECUTIONS = 31

# Function to read a TSP instance from a TSPLIB file, supporting various TSPLIB formats
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
    capacity = None
    node_coords = {}
    edge_weights = []
    problem_type = 'TSP'
    is_symmetric = True
    instance_name = None
    display_data_type = None

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
            elif keyword == 'DISPLAY_DATA_TYPE':
                display_data_type = value
            elif keyword == 'CAPACITY':
                capacity = int(value)
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
            if len(parts) == 3 or len(parts) == 4:
                # Format: <index> <x> <y> (for 2D) or <index> <x> <y> <z> (for 3D)
                index = int(parts[0]) - 1
                x = float(parts[1])
                y = float(parts[2])
                if len(parts) == 4:
                    # 3D coordinate
                    z = float(parts[3])
                    node_coords[index] = (x, y, z)
                else:
                    node_coords[index] = (x, y)
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

    # Construct the distance matrix
    if dimension is None:
        raise ValueError("Dimension not specified in TSPLIB file.")

    # If no edge_weight_type is specified, default to EUC_2D (common case)
    if not edge_weight_type:
        if node_coord_type in ['TWOD_COORDS', 'THREED_COORDS']:
            edge_weight_type = 'EUC_2D' if node_coord_type == 'TWOD_COORDS' else 'EUC_3D'
        else:
            edge_weight_type = 'EXPLICIT'

    # If the problem is EXPLICIT
    if edge_weight_type == 'EXPLICIT' and edge_weight_format and edge_weights:
        distance_matrix = parse_explicit_edge_weights(edge_weights, dimension, edge_weight_format)
    # If the problem has coordinates
    elif edge_weight_type in [
        'EUC_2D', 'EUC_3D', 'MAN_2D', 'MAN_3D', 
        'MAX_2D', 'MAX_3D', 'CEIL_2D', 'GEO', 'ATT'
    ]:
        coords = [node_coords[i] for i in range(dimension)]
        distance_matrix = compute_distance_matrix(coords, edge_weight_type)
    else:
        raise ValueError(f"Unsupported or incomplete handling for EDGE_WEIGHT_TYPE: {edge_weight_type}")

    # Reconstruct cities from node_coords for the solution
    if dimension != len(node_coords):
        raise ValueError("Mismatch in dimension and node coordinates count.")

    cities = [node_coords[i] for i in range(dimension)]
    return instance_name, cities, distance_matrix, is_symmetric

# Function to parse explicit edge weights according to the TSPLIB specification
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
    elif edge_weight_format == 'UPPER_DIAG_ROW':
        # Upper diagonal matrix (row-wise including diagonal entries)
        for i in range(dimension):
            for j in range(i, dimension):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'LOWER_DIAG_ROW':
        # Lower diagonal matrix (row-wise including diagonal entries)
        for i in range(dimension):
            for j in range(0, i + 1):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'UPPER_COL':
        # Upper triangular matrix (column-wise without diagonal)
        for j in range(dimension):
            for i in range(j):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'LOWER_COL':
        # Lower triangular matrix (column-wise without diagonal)
        for j in range(dimension):
            for i in range(j + 1, dimension):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'UPPER_DIAG_COL':
        # Upper diagonal matrix (column-wise including diagonal)
        for j in range(dimension):
            for i in range(j, dimension):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    elif edge_weight_format == 'LOWER_DIAG_COL':
        # Lower diagonal matrix (column-wise including diagonal)
        for j in range(dimension):
            for i in range(0, j + 1):
                distance_matrix[i][j] = edge_weights[idx]
                distance_matrix[j][i] = distance_matrix[i][j]
                idx += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    return distance_matrix

# Function to compute the distance matrix based on the edge weight type
def compute_distance_matrix(coords, edge_weight_type):
    n = len(coords)
    distance_matrix = np.zeros((n, n))

    if edge_weight_type == 'EUC_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist)
    elif edge_weight_type == 'EUC_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(dist)
    elif edge_weight_type == 'MAN_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dist = abs(xi - xj) + abs(yi - yj)
                distance_matrix[i][j] = distance_matrix[j][i] = dist
    elif edge_weight_type == 'MAN_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                dist = abs(xi - xj) + abs(yi - yj) + abs(zi - zj)
                distance_matrix[i][j] = distance_matrix[j][i] = dist
    elif edge_weight_type == 'MAX_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dist = max(abs(xi - xj), abs(yi - yj))
                distance_matrix[i][j] = distance_matrix[j][i] = dist
    elif edge_weight_type == 'MAX_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                dist = max(abs(xi - xj), abs(yi - yj), abs(zi - zj))
                distance_matrix[i][j] = distance_matrix[j][i] = dist
    elif edge_weight_type == 'CEIL_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                distance = math.ceil(dist)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'GEO':
        coords_rad = []
        for (lat, lon) in coords:
            coords_rad.append(convert_geo_coords((lat, lon)))
        for i in range(n):
            lati, loni = coords_rad[i]
            for j in range(i + 1, n):
                latj, lonj = coords_rad[j]
                dist = geo_distance(lati, loni, latj, lonj)
                distance_matrix[i][j] = distance_matrix[j][i] = dist
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
    _min = coord[0] - deg
    latitude = PI * (deg + 5.0 * _min / 3.0) / 180.0
    deg = int(coord[1])
    _min = coord[1] - deg
    longitude = PI * (deg + 5.0 * _min / 3.0) / 180.0
    return latitude, longitude

def geo_distance(lati, loni, latj, lonj):
    RRR = 6378.388
    q1 = math.cos(loni - lonj)
    q2 = math.cos(lati - latj)
    q3 = math.cos(lati + latj)
    dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    return dij

def calcular_longitud(route, distance_matrix):
    total_length = 0
    for i in range(len(route) - 1):
        total_length += distance_matrix[route[i]][route[i + 1]]
    # Add the distance back to the start city to complete the roundtrip
    total_length += distance_matrix[route[-1]][route[0]]
    return total_length

# Genetic Algorithm for TSP
def genetic_algorithm(distance_matrix, max_iterations, seed, run_num, instance_name, population_size, elite_size, mutation_rate):
    random.seed(seed)
    np.random.seed(seed)

    n = len(distance_matrix)
    
    # GA Functions
    def create_initial_population():
        population = []
        for _ in range(population_size):
            individual = list(range(n))
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(route):
        return calcular_longitud(route, distance_matrix)

    def selection(population):
        # Sort by ascending route length (lower is better)
        population.sort(key=lambda x: fitness(x))
        elite_count = int(population_size * elite_size)
        return population[:elite_count]

    def crossover(parent1, parent2):
        start, end = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[start:end + 1] = parent1[start:end + 1]
        idx = 0
        for gene in parent2:
            if gene not in child:
                while child[idx] != -1:
                    idx += 1
                child[idx] = gene
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    population = create_initial_population()
    best_route = None
    best_distance = float('inf')

    # For each iteration, record the best fitness and iteration time
    iteration_results = []
    start_time = time.time()

    # Setup CSV file for iteration results
    results_file_path = f'results/results_{instance_name}_{run_num}.csv'
    os.makedirs('results', exist_ok=True)
    with open(results_file_path, 'w', newline='') as csvfile:
        fieldnames = ['instance_name', 'run_num', 'iteration', 'seed', 'fitness', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    population_current = create_initial_population()

    for iteration in range(1, max_iterations + 1):
        iteration_start_time = time.time()

        selected_parents = selection(population_current)
        next_generation = selected_parents[:]

        # Fill the next generation until we have full population
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            next_generation.append(child)

        population_current = next_generation

        # Evaluate the best individual in the current population
        current_best_distance = float('inf')
        for individual in population_current:
            dist = fitness(individual)
            if dist < current_best_distance:
                current_best_distance = dist

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = population_current[0]

        iteration_time = time.time() - iteration_start_time
        cumulative_time = time.time() - start_time

        # Record iteration data
        iteration_results.append({
            'instance_name': instance_name,
            'run_num': run_num,
            'iteration': iteration,
            'seed': seed,
            'fitness': best_distance,
            'time': cumulative_time
        })

        # Write iteration result to CSV
        with open(results_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['instance_name', 'run_num', 'iteration', 'seed', 'fitness', 'time'])
            writer.writerow(iteration_results[-1])

    return best_route, best_distance, iteration_results

def run_genetic_algorithm(instance_name, distance_matrix, max_iterations, num_runs, population_size, elite_size, mutation_rate):
    # Setup results directories
    os.makedirs('results', exist_ok=True)

    final_statistics = []

    with ProcessPoolExecutor(max_workers=num_runs) as executor:
        seeds = [random.randint(1, 1000000) for _ in range(num_runs)]
        futures = []

        for run_num, seed in enumerate(seeds, start=1):
            futures.append(
                executor.submit(
                    genetic_algorithm, 
                    distance_matrix, 
                    max_iterations, 
                    seed, 
                    run_num, 
                    instance_name,
                    population_size,
                    elite_size,
                    mutation_rate
                )
            )

        results = [f.result() for f in futures]

    for (run_num, seed), (best_route, best_distance, iteration_results) in zip(enumerate(seeds, start=1), results):
        # Compute per-run summary statistics using iteration results (like TS approach)
        fitness_values = [data['fitness'] for data in iteration_results]  # Fitness over all iterations for this run
        time_values = [data['time'] for data in iteration_results]       # Times over all iterations for this run

        min_fitness = min(fitness_values) if fitness_values else float('inf')
        max_fitness = max(fitness_values) if fitness_values else float('-inf')
        avg_fitness = statistics.mean(fitness_values) if fitness_values else float('inf')
        fitness_std_dev = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
        avg_time = statistics.mean(time_values) if time_values else 0

        final_statistics.append({
            'run_num': run_num,
            'final_fitness': best_distance,
            'min_fitness': min_fitness,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'fitness_std_dev': fitness_std_dev,
            'avg_time': avg_time,
            'seed': seed
        })

    return final_statistics

def save_final_statistics(instance_name, final_statistics):
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

def solve_tsp_files(folder_path, file_names):
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name + ".tsp")
        print(f"Solving TSP for {file_name}...\n")

        # Read the TSP instance from the TSPLIB file
        try:
            instance_name, cities, distance_matrix, is_symmetric = read_tsp_instance(file_path)
            if not is_symmetric:
                print(f"Instance {instance_name} is not symmetric. This code only supports symmetric TSP at the moment.")
                continue

            # Run the Genetic Algorithm
            final_statistics = run_genetic_algorithm(
                instance_name,
                distance_matrix,
                GENERATIONS,
                NUM_EXECUTIONS,
                POPULATION_SIZE,
                ELITE_SIZE,
                MUTATION_RATE
            )

            # Save final statistics for this instance
            save_final_statistics(instance_name, final_statistics)

            # Print summary of best found solution across runs
            best_solution_overall = min(final_statistics, key=lambda x: x['final_fitness'])
            print(f"Instance {instance_name} solved. Best overall distance: {best_solution_overall['final_fitness']:.2f}\n")

        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")

if __name__ == '__main__':
    # Example usage
    folder_path = 'tsplib'
    file_names = ["ulysses16", "ulysses22", "gr202", "tsp225", "a280", "pcb442", "gr666"]
    solve_tsp_files(folder_path, file_names)
