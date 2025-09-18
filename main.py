import argparse
import time
import os
import urllib.request
import gzip
import shutil
import pandas as pd

from src.graph import Graph
from graph_loader import load_dimacs_graph, load_snap_graph, get_file_size_mb
from src.bmssp_solver import BmsspSolver, BmsspSolverV2
from src.comparison_solvers import dijkstra, bellman_ford

# A dictionary to manage the configurations for different datasets.
# Each entry specifies how to load the data, its filename, properties,
# and sample nodes for testing the shortest path algorithms.
DATASETS = {
    "rome": {
        "loader": load_dimacs_graph,
        "filename": "Rome99.txt",
        "is_directed": True,
        "start_node": 6,
        "end_node": 1778
    },
    "livejournal": {
        "url": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
        "loader": load_snap_graph,
        "filename": "soc-LiveJournal1.txt",
        "is_directed": True,
        "start_node": 1,
        "end_node": 1000
    },
    "pokec": {
        "url": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
        "loader": load_snap_graph,
        "filename": "soc-pokec-relationships.txt",
        "is_directed": True,
        "start_node": 1452585,
        "end_node": 1618281
    },
    "california": {
        "url": "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-CA.txt",
        "is_directed": False,
        "start_node": 807041,
        "end_node": 1453117
    },
    "pennsylvania": {
        "url": "https://snap.stanford.edu/data/roadNet-PA.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-PA.txt",
        "is_directed": False,
        "start_node": 853045,
        "end_node": 992376
    },
    "texas": {
        "url": "https://snap.stanford.edu/data/roadNet-TX.txt.gz",
        "loader": load_snap_graph,
        "filename": "roadNet-TX.txt",
        "is_directed": False,
        "start_node": 558629,
        "end_node": 613982
    },
    "google": {
        "url": "https://snap.stanford.edu/data/web-Google.txt.gz",
        "loader": load_snap_graph,
        "filename": "web-Google.txt",
        "is_directed": True,
        "start_node": 895428,
        "end_node": 228498,
        "note": "Web graph - may have disconnected components"
    },
    "stanford": {
        "url": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
        "loader": load_snap_graph,
        "filename": "web-Stanford.txt",
        "is_directed": True,
        "start_node": 235899,
        "end_node": 23074,
        "note": "Web graph - paths may not exist between arbitrary nodes due to disconnected components"
    },
}

# Define the directory where all graph data files will be stored.
DATA_DIR = "data"

def prepare_dataset(dataset_name: str, dataset_info: dict):
    """
    Ensures the required dataset file is available in the 'data' directory.
    If a 'url' is provided and the file is missing, it will be downloaded and extracted.
    If no 'url' is given, it assumes the file is local and checks for its existence.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    final_path = os.path.join(DATA_DIR, dataset_info["filename"])
    
    if os.path.exists(final_path):
        print(f"Dataset '{dataset_name}' found locally at '{final_path}'.")
        return final_path

    if "url" not in dataset_info:
        print(f"Error: Local dataset file '{final_path}' not found.")
        print("Please ensure the file is placed in the 'data' directory.")
        return None

    url = dataset_info["url"]
    compressed_filename = os.path.basename(url)
    compressed_path = os.path.join(DATA_DIR, compressed_filename)

    print(f"Downloading dataset '{dataset_name}' from {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(compressed_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(compressed_path): os.remove(compressed_path)
        return None

    if url.endswith(".gz"):
        print(f"Extracting {compressed_path}...")
        try:
            with gzip.open(compressed_path, 'rb') as f_in, open(final_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(compressed_path)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting {compressed_path}: {e}")
            if os.path.exists(final_path): os.remove(final_path)
            return None
    
    return final_path




def run_benchmark(dataset_name: str, solver_version: str, use_cache: bool = True):
    """
    Loads a specified graph and runs a comparative benchmark of the SSSP algorithms.
    """
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' is not defined.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return

    dataset_info = DATASETS[dataset_name]
    
    filepath = prepare_dataset(dataset_name, dataset_info)
    if not filepath:
        print("Failed to prepare dataset. Aborting benchmark.")
        return

    # Show file size for large files
    file_size_mb = get_file_size_mb(filepath)
    if file_size_mb > 10:
        print(f"Loading large file ({file_size_mb:.1f} MB) - this may take a moment...")
    
    print(f"\nLoading graph from {filepath}...")
    try:
        loader_func = dataset_info["loader"]
        if loader_func == load_snap_graph:
            graph = loader_func(filepath, is_directed=dataset_info.get("is_directed", True), use_cache=use_cache)
        else:
            graph = loader_func(filepath, use_cache=use_cache)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    source_node = dataset_info["start_node"]
    goal_node = dataset_info["end_node"]

    source_idx = source_node - 1 if dataset_name == "rome" else source_node
    goal_idx = goal_node - 1 if dataset_name == "rome" else goal_node


    if not (0 <= source_idx < graph.vertices and 0 <= goal_idx < graph.vertices):
        print(f"Error: Source ({source_idx}) or Goal ({goal_idx}) node is out of bounds for the graph (0 to {graph.vertices-1}).")
        return
    
    print(f"\nFinding shortest path from node {source_node} to {goal_node}")
    print("-" * 50)

    # Execute and time the selected BMSSP solver.
    print(f"\nRunning BMSSP Algorithm (Version: {solver_version})...")
    if solver_version == 'v2':
        solver = BmsspSolverV2(graph)
    else:
        solver = BmsspSolver(graph)

    start_time = time.time()
    bmssp_result = solver.solve(source_idx, goal_idx)
    end_time = time.time()
    
    if bmssp_result:
        distance, path = bmssp_result
        print(f"✅ BMSSP ({solver_version.upper()}) Result: Distance = {distance:.2f}, Path length = {len(path)} nodes")
    else:
        print(f"❌ BMSSP ({solver_version.upper()}): No path found.")
    print(f"   Execution time: {end_time - start_time:.4f} seconds")

    # Execute and time Dijkstra's algorithm for a performance comparison.
    print("\nRunning Dijkstra's Algorithm for comparison...")
    start_time = time.time()
    
    dijkstra_result = dijkstra(graph, source_idx, goal_idx)
    end_time = time.time()

    if dijkstra_result:
        distance, path = dijkstra_result
        print(f"✅ Dijkstra Result: Distance = {distance:.2f}, Path length = {len(path)} nodes")
    else:
        print("❌ Dijkstra: No path found.")
    print(f"   Execution time: {end_time - start_time:.4f} seconds")

    # Execute Bellman-Ford only on smaller graphs.
    if graph.vertices < 50000:
        print("\nRunning Bellman-Ford Algorithm for comparison...")
        start_time = time.time()
        bf_dist, bf_path, bf_cycle = bellman_ford(graph, source_idx, goal_idx)
        end_time = time.time()

        if bf_dist is not None:
            print(f"✅ Bellman-Ford Result: Distance = {bf_dist:.2f}, Path length = {len(bf_path)} nodes")
            if bf_cycle:
                print("   Warning: A negative-weight cycle was detected.")
        else:
            print("❌ Bellman-Ford: No path found.")
        print(f"   Execution time: {end_time - start_time:.4f} seconds")
    else:
        print("\nSkipping Bellman-Ford for large graph to save time.")
        
    print("-" * 50)


def main():
    """
    Parses command-line arguments to select a dataset and solver, then runs the benchmark.
    """
    parser = argparse.ArgumentParser(description="Run shortest path algorithms on various datasets.")
    parser.add_argument(
        '--data', 
        type=str, 
        default='rome',
        choices=DATASETS.keys(),
        help=f"The dataset to use for the benchmark. Defaults to 'rome'."
    )
    parser.add_argument(
        '--solver',
        type=str,
        default='v2',
        choices=['v1', 'v2'],
        help="The BMSSP solver version to use. 'v1' is the basic implementation, 'v2' is the optimized version. Defaults to 'v2'."
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help="Disable caching for this run (will still load from cache if available)."
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help="Force reload from original data files, bypassing cache completely."
    )
    args = parser.parse_args()
    
    if args.force_reload:
        use_cache = False
    else:
        use_cache = not args.no_cache
    run_benchmark(args.data, args.solver, use_cache)


if __name__ == "__main__":
    main()