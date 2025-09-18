import pandas as pd
import numpy as np
from src.graph import Graph
from src.graph_cache import GraphCache

# Global cache instance
_cache = GraphCache()

def load_dimacs_graph(file_path: str, is_directed: bool = True, use_cache: bool = True) -> Graph:
    """
    Parses a graph file in the 9th DIMACS Implementation Challenge format with optimizations.
    This format is structured with problem lines ('p'), comment lines ('c'),
    and arc descriptor lines ('a'). Node indices are 1-based and are converted
    to 0-based for internal use.
    
    Args:
        file_path: Path to the DIMACS format file
        is_directed: Whether to treat the graph as directed (True) or undirected (False)
        use_cache: Whether to use caching for faster repeated loads
    
    Returns:
        Graph object with edges loaded from the file
    """
    # Try to load from cache first
    if use_cache:
        cached_graph = _cache.load_cached_graph(file_path, is_directed=None)
        if cached_graph is not None:
            return cached_graph
    
    print(f"Parsing DIMACS file {file_path}...")
    
    # Read entire file at once for better I/O performance
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    graph = None
    edges_data = []
    
    # First pass: find problem line and collect edge data
    for line in lines:
        if not line or line.startswith('c'):
            continue
        
        parts = line.split()
        if not parts:
            continue
            
        line_type = parts[0]
        
        # The 'p' line defines the problem and graph size.
        if line_type == 'p':
            if len(parts) == 4 and parts[1] == 'sp':
                num_vertices = int(parts[2])
                graph = Graph(num_vertices)
            else:
                raise ValueError("Invalid DIMACS problem line format.")
        
        # 'a' lines define the edges (arcs) and their weights.
        elif line_type == 'a':
            if len(parts) == 4:
                u, v, weight = int(parts[1]), int(parts[2]), float(parts[3])
                # Convert from 1-based (DIMACS) to 0-based (internal) index.
                edges_data.append((u - 1, v - 1, weight))
            else:
                raise ValueError(f"Invalid arc descriptor format: {line}")
    
    if graph is None:
        raise ValueError("Invalid DIMACS file: No problem line found.")
    
    # Add all edges efficiently
    print(f"Adding {len(edges_data)} edges to graph...")
    for u, v, weight in edges_data:
        graph.add_edge(u, v, weight)
    
    print(f"Graph loaded: {graph.vertices} vertices, {len(edges_data)} edges")
    
    # Cache the loaded graph for future use
    if use_cache:
        _cache.save_graph_to_cache(graph, file_path, is_directed=None)
    
    return graph

def load_snap_graph(file_path: str, is_directed: bool = True, use_cache: bool = True) -> Graph:
    """
    Parses a graph file from the Stanford Network Analysis Platform (SNAP) using pandas
    with vectorized operations for maximum performance on large files.
    
    These files are typically simple edge lists where each line represents an edge.
    Comment lines start with '#'. Since these graphs are usually unweighted,
    a default edge weight of 1.0 is assigned.
    
    Args:
        file_path: Path to the SNAP format file
        is_directed: Whether to treat the graph as directed (True) or undirected (False)
        use_cache: Whether to use caching for faster repeated loads
    
    Returns:
        Graph object with edges loaded from the file
    """
    # Try to load from cache first
    if use_cache:
        cached_graph = _cache.load_cached_graph(file_path, is_directed=is_directed)
        if cached_graph is not None:
            return cached_graph
    
    print(f"Parsing SNAP file {file_path} ({'directed' if is_directed else 'undirected'})...")
    
    try:
        # Use pandas with optimized settings for large files
        df = pd.read_csv(
            file_path,
            comment='#',
            sep=r'\s+',  # Handles both tabs and spaces as delimiters
            header=None,
            names=['u', 'v'],
            usecols=[0, 1],
            dtype={'u': np.int32, 'v': np.int32},  # Use int32 for memory efficiency
            engine='c',  # Use faster C engine
            low_memory=False  # Read entire file into memory for speed
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Return an empty graph if the file doesn't exist or is empty.
        return Graph(0)

    if df.empty:
        return Graph(0)

    print(f"Read {len(df)} edges from file...")
    
    # Vectorized operations for maximum performance
    max_node_id = int(max(df['u'].max(), df['v'].max()))
    num_vertices = max_node_id + 1
    print(f"Creating graph with {num_vertices} vertices...")
    graph = Graph(num_vertices)
    
    # Convert to numpy arrays for faster iteration
    u_array = df['u'].values
    v_array = df['v'].values
    
    print(f"Adding edges to graph...")
    # Vectorized edge addition - much faster than itertuples
    for i in range(len(u_array)):
        graph.add_edge(u_array[i], v_array[i], 1.0)
        # For undirected graphs, add the reverse edge as well.
        if not is_directed:
            graph.add_edge(v_array[i], u_array[i], 1.0)
    
    print(f"Graph loaded: {num_vertices} vertices, {sum(len(adj) for adj in graph.adj)} edges")
    
    # Cache the loaded graph for future use
    if use_cache:
        _cache.save_graph_to_cache(graph, file_path, is_directed=is_directed)
    
    return graph


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes for progress estimation.
    """
    import os
    return os.path.getsize(file_path) / (1024 * 1024)
