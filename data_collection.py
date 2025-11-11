import os
import csv
import argparse
import subprocess
import xml.etree.ElementTree as ET

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import traci
from sumolib import checkBinary
from sumolib.miscutils import getFreeSocketPort

import multiprocessing as mp

def get_dense_grid_network(grid_number, grid_length, network_file):
    # No edge removal: make fully connected grid
    if os.path.exists(network_file):
        print(f"File {network_file} already exists. Use existing file.")
    else:
        subprocess.call(f'netgenerate --grid --grid.number {grid_number} --grid.length {grid_length} --output-file {network_file}', shell=True)

    # Need to reconstruct edge naming for routes
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    start_edges_up, start_edges_down, start_edges_left, start_edges_right = [], [], [], []
    end_edges_up, end_edges_down, end_edges_left, end_edges_right = [], [], [], []
    for i in range(grid_number):
        if i == 0:
            for j in range(grid_number):
                if j != 0 and j != grid_number - 1:
                    start_edges_left.append(f"{alphabet[i]}{j}{alphabet[i+1]}{j}")
                    end_edges_left.append(f"{alphabet[i+1]}{j}{alphabet[i]}{j}")
        elif i == grid_number - 1:
            for j in range(grid_number):
                if j != 0 and j != grid_number - 1:
                    start_edges_right.append(f"{alphabet[i]}{j}{alphabet[i-1]}{j}")
                    end_edges_right.append(f"{alphabet[i-1]}{j}{alphabet[i]}{j}")
        else:
            start_edges_up.append(f"{alphabet[i]}0{alphabet[i]}1")
            end_edges_up.append(f"{alphabet[i]}1{alphabet[i]}0")
            start_edges_down.append(f"{alphabet[i]}{grid_number-1}{alphabet[i]}{grid_number-2}")
            end_edges_down.append(f"{alphabet[i]}{grid_number-2}{alphabet[i]}{grid_number-1}")
    start_edges = [start_edges_up, start_edges_down, start_edges_left, start_edges_right]
    end_edges = [end_edges_up, end_edges_down, end_edges_left, end_edges_right]
    return start_edges, end_edges

def randomize_routes(route_number, route_type, min_num_vehicles, max_num_vehicles, simulation_time, 
                     start_edges, end_edges, route_file, seed=42):
    # Randomize as controlled by the idx/seed
    random.seed(seed)
    np.random.seed(seed)
    routes = []
    for _ in range(route_number):
        if route_type == "random":
            i, j = np.random.choice(4, 2, replace=False)
            start_edge = np.random.choice(start_edges[i])
            end_edge = np.random.choice(end_edges[j])
            num_vehicles = np.random.randint(min_num_vehicles, max_num_vehicles)
            routes.append((start_edge, end_edge, num_vehicles))
        elif route_type == "straight":
            i = np.random.choice(4)
            if i < 2:
                j = (i + 1) % 2
            else:
                j = (i + 1) % 2 + 2
            idx = np.random.choice(len(start_edges[i]))
            start_edge = start_edges[i][idx]
            end_edge = end_edges[j][idx]
            num_vehicles = np.random.randint(min_num_vehicles, max_num_vehicles)
            routes.append((start_edge, end_edge, num_vehicles))
        elif route_type == "major_left":
            if random.random() < 0.5:
                start_edge = start_edges[0][len(start_edges[0]) // 2]
                end_edge = end_edges[1][len(end_edges[1]) // 2]
                routes.append((start_edge, end_edge, max_num_vehicles))
            else:
                i, j = np.random.choice(4, 2, replace=False)
                start_edge = np.random.choice(start_edges[i])
                end_edge = np.random.choice(end_edges[j])
                routes.append((start_edge, end_edge, min_num_vehicles))
    with open(f"{route_file}", 'w') as f:
        f.write(f'<routes>\n')
        for i, (start_edge, end_edge, num_vehicles) in enumerate(routes):
            f.write(f'    <flow id="{i}" begin="0" end="{simulation_time // 2}" from="{start_edge}" to="{end_edge}" vehsPerHour="{num_vehicles}" />\n')
        f.write('</routes>\n')
    # The list of routes can be used as a description of "X" for input features
    # We'll encode the route configuration in a reversible way
    return routes

def simulation(idx, args, start_edges, end_edges, folder_name, seed):
    # Randomize the route file each iteration
    route_file = f"{folder_name}/gen_{idx}.rou.xml"
    routes = randomize_routes(
        args.route_number, args.route_type, args.min_num_vehicles, args.max_num_vehicles, args.simulation_time,
        start_edges, end_edges, route_file, seed=seed+idx
    )

    # Write the configuration file (use the fixed network!)
    sumocfg_file = f"{folder_name}/gen_{idx}.sumocfg"
    tripinfo_file = f"{folder_name}/gen_{idx}.tripinfo.xml"
    with open(sumocfg_file, "w") as f:
        f.write(f'<configuration>\n')
        f.write(f'    <input>\n')
        f.write(f'        <net-file value="default.net.xml"/>\n')
        f.write(f'        <route-files value="gen_{idx}.rou.xml"/>\n')
        f.write(f'    </input>\n')
        f.write(f'    <time>\n')
        f.write(f'        <begin value="0"/>\n')
        f.write(f'        <end value="{args.simulation_time}"/>\n')
        f.write(f'    </time>\n')
        f.write(f'</configuration>\n')
    if args.visualize:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumo_cmd = [sumoBinary, "-c", sumocfg_file, "--no-warnings", "--no-step-log", "--seed", "42", "--tripinfo-output", tripinfo_file]
    port = getFreeSocketPort()
    traci.start(sumo_cmd, port=port)
    for _ in range(args.simulation_time):
        traci.simulationStep()
    traci.close()

    # Post-process
    waiting_time_list = []
    traveling_time_list = []
    last_vehicle_arrival_time = 0
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    for child in root:
        waiting_time_list.append(float(child.attrib['waitingTime']))
        traveling_time_list.append(float(child.attrib['duration']))
        last_vehicle_arrival_time = max(last_vehicle_arrival_time, float(child.attrib['arrival']))
    y = np.array([np.mean(waiting_time_list), np.mean(traveling_time_list), last_vehicle_arrival_time])
    print(f"Average waiting time: {np.mean(waiting_time_list):.2f}", end="\t")
    print(f"Average traveling time: {np.mean(traveling_time_list):.2f}", end="\t")
    print(f"Last vehicle arrival time: {last_vehicle_arrival_time:.2f}")

    # Encode the route configuration as a string for "X"
    route_strs = [f"{start}-{end}:{num}" for (start, end, num) in routes]
    X = '|'.join(route_strs)
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Network parameters
    parser.add_argument("--grid_number", type=int, default=12)
    parser.add_argument("--grid_length", type=float, default=50.0)
    
    # Route parameters
    parser.add_argument("--route_number", type=int, default=20)
    parser.add_argument("--route_type", type=str, default="random")
    parser.add_argument("--min_num_vehicles", type=int, default=100)
    parser.add_argument("--max_num_vehicles", type=int, default=200)
    
    # Simulation parameters
    parser.add_argument("--simulation_time", type=int, default=1800)
    
    # seed
    parser.add_argument("--seed", type=int, default=42)
    
    # visualize
    parser.add_argument("--visualize", action="store_true")
    
    # data collection
    parser.add_argument("--num_data_points", type=int, default=10)
    
    args = parser.parse_args()
    
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    
    folder_name = f"sumo/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    # Get (and possibly generate) fixed dense network
    network_file = f"{folder_name}/default.net.xml"
    start_edges, end_edges = get_dense_grid_network(args.grid_number, args.grid_length, network_file)
    
    # parallel simulation over randomized routes
    inputs = [(i, args, start_edges, end_edges, folder_name, seed) for i in range(args.num_data_points)]
    with mp.Pool(8) as pool:
        data = list(pool.starmap(simulation, tqdm(inputs, total=args.num_data_points)))
        
    # Save data
    # For randomized routes, X is a route encoding string.
    X = [d[0] for d in data]
    y = np.stack([d[1] for d in data])
    save_path = f"results/N{args.grid_number}_L{args.grid_length}/R{args.route_number}_T{args.route_type}_MIN{args.min_num_vehicles}_MAX{args.max_num_vehicles}/data/preprocessed"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    npz_save_path = os.path.join(save_path, f"data_{args.num_data_points}.npz")
    np.savez_compressed(npz_save_path, X=X, y=y)
    
    csv_save_path = os.path.join(save_path, f"sumo_preprocessed_dataset_iter1.csv")
    csv_data = []
    data_loaded = np.load(npz_save_path, allow_pickle=True)
    data_x = data_loaded["X"]
    data_y = data_loaded["y"]
    
    for x, yrow in zip(data_x, data_y):
        route_encoding = x  # already a string
        waiting_time = yrow[0]
        csv_data.append([route_encoding, waiting_time])
    
    with open(csv_save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["route_configuration", "waiting_time"])
        writer.writerows(csv_data)