import os
import csv
import glob
import random
import time
import statistics
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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
        elif edge_data_section:
            if line.startswith('EOF'):
                # End of edge data section
                edge_data_section = False
                break
            # TODO: If needed, implement parsing edge data
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
    elif edge_weight_type in ['EUC_2D', 'EUC_3D', 'MAN_2D', 'MAN_3D', 'MAX_2D', 'MAX_3D', 'CEIL_2D', 'GEO', 'ATT']:
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
                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(distance)
    elif edge_weight_type == 'EUC_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
                distance_matrix[i][j] = distance_matrix[j][i] = round(distance)
    elif edge_weight_type == 'MAN_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                distance = abs(xi - xj) + abs(yi - yj)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'MAN_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                distance = abs(xi - xj) + abs(yi - yj) + abs(zi - zj)
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'MAX_2D':
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                distance = max(abs(xi - xj), abs(yi - yj))
                distance_matrix[i][j] = distance_matrix[j][i] = distance
    elif edge_weight_type == 'MAX_3D':
        for i in range(n):
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                xj, yj, zj = coords[j]
                distance = max(abs(xi - xj), abs(yi - yj), abs(zi - zj))
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
        # Convert coordinates from degrees to radians
        coords_rad = []
        for (lat, lon) in coords:
            coords_rad.append(convert_geo_coords((lat, lon)))
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

# Function to convert geographical coordinates in DDD.MM format to radians as per TSPLIB
def convert_geo_coords(coord):
    PI = 3.141592653589793
    deg = int(coord[0])
    _min = coord[0] - deg
    latitude = PI * (deg + 5.0 * _min / 3.0) / 180.0
    deg = int(coord[1])
    _min = coord[1] - deg
    longitude = PI * (deg + 5.0 * _min / 3.0) / 180.0
    return latitude, longitude

# Function to calculate the geographical distance according to TSPLIB
def geo_distance(lati, loni, latj, lonj):
    RRR = 6378.388
    q1 = math.cos(loni - lonj)
    q2 = math.cos(lati - latj)
    q3 = math.cos(lati + latj)
    dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    return dij

# Calculate the total length of a route using the distance matrix
def calcular_longitud(solution, distance_matrix):
    total_length = 0
    for i in range(len(solution) - 1):
        total_length += distance_matrix[solution[i]][solution[i + 1]]
    # Add the distance back to the start city to complete the roundtrip
    total_length += distance_matrix[solution[-1]][solution[0]]
    return total_length

# Tabu Search Algorithm for TSP
def tabu_search(cities, distance_matrix, max_iterations, tabu_size, tabu_tenure, iteration_seed, run_num, instance_name):
    random.seed(iteration_seed)
    n = len(cities)
    current_solution = list(range(n))
    random.shuffle(current_solution)
    tabu_list = {}  # Using a dictionary to store tabu moves and their tenures for O(1) access

    best_solution = current_solution[:]
    best_length = calcular_longitud(best_solution, distance_matrix)

    iteration = 0
    start_time = time.time()

    fitness_history = []
    time_history = []
    cumulative_time = 0.0

    # Limit on the number of neighbors we generate each iteration to speed up computations
    # Instead of generating all O(n^2) neighbors, we only generate a subset for evaluation.
    NEIGHBOR_LIMIT = 100  # You can adjust this

    while iteration < max_iterations:
        iteration_start_time = time.time()

        # Generate random neighbors by swapping two cities
        neighbors = []
        neighbors_moves = []
        for _ in range(NEIGHBOR_LIMIT):
            i_idx, j_idx = random.sample(range(n), 2)
            move = (min(i_idx, j_idx), max(i_idx, j_idx))
            if move not in tabu_list:
                neighbor = current_solution[:]
                neighbor[i_idx], neighbor[j_idx] = neighbor[j_idx], neighbor[i_idx]
                neighbors.append(neighbor)
                neighbors_moves.append(move)

        # Evaluate neighbor solutions
        neighbors_lengths = [calcular_longitud(neighbor, distance_matrix) for neighbor in neighbors]

        # Update the tabu list (decrementing tenure)
        tabu_list = {move: tenure - 1 for move, tenure in tabu_list.items() if tenure > 1}

        if neighbors:
            best_neighbor_index = np.argmin(neighbors_lengths)
            best_neighbor = neighbors[best_neighbor_index]
            best_neighbor_length = neighbors_lengths[best_neighbor_index]
            best_neighbor_move = neighbors_moves[best_neighbor_index]

            # If the best neighbor improves on the best known solution, update it
            if best_neighbor_length < best_length:
                best_solution = best_neighbor
                best_length = best_neighbor_length

            # Add current move to the tabu list with a certain tenure
            tabu_list[best_neighbor_move] = tabu_tenure

            # Move to best neighbor solution
            current_solution = best_neighbor[:]

        iteration += 1

        iteration_time = time.time() - iteration_start_time
        cumulative_time += iteration_time
        fitness_history.append(best_length)
        time_history.append(iteration_time)

        # Save iteration results
        with open(f'results/results_{instance_name}_{run_num}.csv', 'a', newline='') as csvfile:
            fieldnames = ['instance_name', 'run_num', 'iteration', 'seed', 'length', 'tabu_size', 'tabu_tenure', 'iteration_time', 'cumulative_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({
                'instance_name': instance_name,
                'run_num': run_num,
                'iteration': iteration,
                'seed': iteration_seed,
                'length': best_length,
                'tabu_size': tabu_size,
                'tabu_tenure': tabu_tenure,
                'iteration_time': iteration_time,
                'cumulative_time': cumulative_time
            })

    avg_fitness = sum(fitness_history) / len(fitness_history) if fitness_history else float('inf')
    avg_iteration_time = sum(time_history) / len(time_history) if time_history else 0
    min_fitness = min(fitness_history) if fitness_history else float('inf')
    max_fitness = max(fitness_history) if fitness_history else 0
    fitness_std_dev = statistics.stdev(fitness_history) if len(fitness_history) > 1 else 0
    final_fitness = best_length  # The final best solution found in this run

    # Save final statistics for this run
    with open(f'results/statistics_{instance_name}.csv', 'a', newline='') as stats_csvfile:
        fieldnames = [
            'instance_name', 'run_num', 'final_fitness', 'min_fitness', 
            'max_fitness', 'avg_fitness', 'fitness_std_dev', 'avg_iteration_time'
        ]
        stats_writer = csv.DictWriter(stats_csvfile, fieldnames=fieldnames)
        if stats_csvfile.tell() == 0:
            stats_writer.writeheader()
        stats_writer.writerow({
            'instance_name': instance_name,
            'run_num': run_num,
            'final_fitness': final_fitness,
            'min_fitness': min_fitness,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'fitness_std_dev': fitness_std_dev,
            'avg_iteration_time': avg_iteration_time
        })

    return best_solution, best_length

# Function to run Tabu Search
def run_tabu_search(instances, num_runs, max_iterations, tabu_size, tabu_tenure):
    os.makedirs('results', exist_ok=True)

    for instance_path in instances:
        # Get the instance name
        instance_filename = os.path.splitext(os.path.basename(instance_path))[0]
        
        print(f"Processing instance {instance_filename} from {instance_path}")

        # Read the TSP instance from the TSPLIB file
        try:
            instance_name, cities, distance_matrix, is_symmetric = read_tsp_instance(instance_path)
            if not is_symmetric:
                print(f"Instance {instance_name} is not symmetric. This code only supports symmetric TSP at the moment.")
                continue
        except Exception as e:
            print(f"Error processing {instance_filename}: {e}")
            continue

        with ProcessPoolExecutor(max_workers=num_runs) as executor:
            unique_seeds = [random.randint(1, 1000000) for _ in range(num_runs)]
            results = list(executor.map(
                tabu_search,
                [cities] * num_runs,
                [distance_matrix] * num_runs,
                [max_iterations] * num_runs,
                [tabu_size] * num_runs,
                [tabu_tenure] * num_runs,
                unique_seeds,
                range(1, num_runs + 1),
                [instance_name] * num_runs
            ))

        for i, (best_solution, best_length) in enumerate(results, 1):
            print(f"Instance {instance_name}, Run {i}: Best found solution with distance {best_length}")
            print("=" * 50)

if __name__ == "__main__":
    # Specific files to be processed
    required_files = {"ulysses16", "ulysses22", "gr202", "tsp225", "a280", "pcb442", "gr666"}
    # Get the list of .tsp files in the directory tsplib
    tsp_directory = 'tsplib'
    tsp_files = []
    for root, dirs, files in os.walk(tsp_directory):
        for file in files:
            # Filter only the files that are in the required_files set
            if file.endswith('.tsp') and os.path.splitext(file)[0] in required_files:
                tsp_files.append(os.path.join(root, file))

    num_runs = 31
    max_iterations = 5000
    tabu_size = 100
    tabu_tenure = 512

    run_tabu_search(tsp_files, num_runs, max_iterations, tabu_size, tabu_tenure)
