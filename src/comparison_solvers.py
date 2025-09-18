import heapq
from typing import Optional, Tuple, List
from .graph import Graph

INFINITY = float('inf')

def dijkstra(graph: Graph, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    """
    Classic Dijkstra's algorithm for single-source shortest path with non-negative weights.
    
    This is the gold standard shortest path algorithm for graphs with non-negative edge weights,
    serving as the baseline comparison for BMSSP performance evaluation. Dijkstra's algorithm
    is optimal in the sense that it examines each vertex exactly once and always finds the
    true shortest path.
    
    HISTORICAL CONTEXT:
    Developed by Edsger Dijkstra in 1956, this algorithm was one of the first efficient
    solutions to the shortest path problem and remains widely used today. Its elegance
    lies in the greedy approach: always process the "closest" unvisited vertex.
    
    ALGORITHM PRINCIPLE:
    The algorithm maintains a "cloud" of vertices for which shortest paths are known.
    It repeatedly:
    1. Selects the unvisited vertex with minimum distance (greedy choice)
    2. Adds it to the cloud (shortest path is now finalized)
    3. Relaxes all edges from this vertex to update neighbor distances
    
    CORRECTNESS PROOF SKETCH:
    - Invariant: All vertices in the cloud have correct shortest path distances
    - When we select vertex u with minimum distance d[u], any path to u through
      unvisited vertices would have distance ≥ d[u] (since all edge weights ≥ 0)
    - Therefore d[u] is indeed the shortest path distance to u
    
    PERFORMANCE CHARACTERISTICS:
    - Time Complexity: O((V + E) log V) with binary heap
    - Space Complexity: O(V) for distance array and priority queue
    - Practical performance: Excellent on most real-world graphs
    
    COMPARISON WITH BMSSP:
    - Dijkstra: O(m log n), examines vertices in perfect distance order
    - BMSSP: O(m log^(2/3) n), examines vertices in approximate distance order
    - Trade-off: BMSSP sacrifices perfect ordering for better asymptotic complexity
    
    IMPLEMENTATION DETAILS:
    - Uses Python's heapq (binary min-heap) for priority queue
    - Implements early termination when goal is reached
    - Handles duplicate entries in priority queue (lazy deletion)
    - Reconstructs path using predecessor array
    
    Args:
        graph: Input graph with non-negative edge weights
        source: Starting vertex index (0-based)
        goal: Target vertex index (0-based)
        
    Returns:
        Tuple of (distance, path) if path exists, None otherwise
        Path is list of vertex indices from source to goal
        
    Raises:
        No exceptions - handles invalid inputs gracefully by returning None
    """
    # STEP 1: INITIALIZATION
    # Initialize data structures for the algorithm
    distances = [INFINITY] * graph.vertices  # Shortest known distance to each vertex
    predecessors = [None] * graph.vertices   # Previous vertex on shortest path (for reconstruction)
    distances[source] = 0.0                  # Source has distance 0 to itself
    
    # STEP 2: PRIORITY QUEUE SETUP
    # The priority queue stores (distance, vertex_id) tuples
    # Python's heapq implements a min-heap, automatically giving us minimum distance vertex
    pq = [(0.0, source)]  # Start with source vertex at distance 0
    
    # STEP 3: MAIN ALGORITHM LOOP
    # Continue until priority queue is empty or goal is reached
    while pq:
        # Extract vertex with minimum distance (greedy choice)
        current_dist, u = heapq.heappop(pq)

        # OPTIMIZATION: Skip stale entries (lazy deletion)
        # This handles the case where we've found a better path to vertex u
        # since this entry was added to the priority queue
        if current_dist > distances[u]:
            continue
        
        # EARLY TERMINATION: Stop when goal is reached
        # Since we process vertices in distance order, first time we reach
        # the goal, we've found the shortest path
        if u == goal:
            break

        # STEP 4: EDGE RELAXATION
        # Examine all neighbors of the current vertex
        for edge in graph.adj[u]:
            # Calculate potential distance improvement through current vertex
            new_dist = distances[u] + edge.weight
            
            # RELAXATION TEST: Update if we found a shorter path
            if new_dist < distances[edge.to]:
                # Update shortest distance and predecessor
                distances[edge.to] = new_dist
                predecessors[edge.to] = u
                
                # Add to priority queue for future processing
                # Note: We don't remove old entries (lazy deletion approach)
                heapq.heappush(pq, (new_dist, edge.to))

    # STEP 5: REACHABILITY CHECK
    # If goal distance is still infinity, no path exists
    if distances[goal] == INFINITY:
        return None

    # STEP 6: PATH RECONSTRUCTION
    # Backtrack from goal to source using predecessor array
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        if curr == source:  # Successfully reached source
            break
        curr = predecessors[curr]
    
    # Validate that we successfully reconstructed a complete path
    if not path or path[-1] != source:
        return None

    # Return distance and path (reverse path since we built it backwards)
    return distances[goal], path[::-1]


def bellman_ford(graph: Graph, source: int, goal: int) -> Tuple[Optional[float], List[int], bool]:
    """
    Bellman-Ford algorithm for shortest paths with negative edge weights and cycle detection.
    
    This algorithm is more general than Dijkstra's algorithm as it can handle negative edge weights
    and detect negative-weight cycles. While slower than Dijkstra's O(m log n), it provides
    important theoretical foundations and serves as a comparison point for understanding
    the limitations and advantages of different shortest path approaches.
    
    HISTORICAL SIGNIFICANCE:
    - Developed independently by Richard Bellman (1955) and Lester Ford Jr. (1956)
    - Predates Dijkstra's algorithm and handles more general case
    - Foundation for many other algorithms (SPFA, distance-vector routing protocols)
    
    ALGORITHM PRINCIPLE:
    The algorithm is based on the principle of relaxation and dynamic programming:
    - After i iterations, we have found all shortest paths with at most i edges
    - Since shortest paths in a graph with n vertices have at most n-1 edges,
      n-1 iterations are sufficient to find all shortest paths
    - A final iteration detects negative cycles
    
    THEORETICAL FOUNDATION:
    - Optimal substructure: Shortest paths have shortest subpaths
    - Overlapping subproblems: Same distance computations repeated
    - Dynamic programming: Build solution incrementally
    
    KEY DIFFERENCES FROM DIJKSTRA:
    
    1. EDGE WEIGHTS: Handles negative weights (Dijkstra requires non-negative)
    2. APPROACH: Processes all edges repeatedly (Dijkstra processes vertices once)
    3. COMPLEXITY: O(VE) time (Dijkstra is O((V+E) log V))
    4. CYCLE DETECTION: Detects negative cycles (Dijkstra cannot handle them)
    5. ORDER: No specific vertex processing order (Dijkstra uses priority queue)
    
    ALGORITHM PHASES:
    
    PHASE 1: RELAXATION (V-1 iterations)
    - Relax all edges in each iteration
    - After i iterations, shortest paths with ≤ i edges are correct
    - Guarantees correctness after V-1 iterations (no negative cycles)
    
    PHASE 2: NEGATIVE CYCLE DETECTION
    - One additional iteration to check for improvements
    - If any distance can still be reduced, negative cycle exists
    - Critical for correctness in graphs with negative weights
    
    PRACTICAL APPLICATIONS:
    - Network routing protocols (distance-vector algorithms)
    - Currency arbitrage detection (negative cycles = profit opportunities)
    - Game theory and economics (negative utility cycles)
    - Constraint satisfaction problems
    
    PERFORMANCE CHARACTERISTICS:
    - Time Complexity: O(VE) - can be slow for dense graphs
    - Space Complexity: O(V) for distance and predecessor arrays
    - Practical performance: Much slower than Dijkstra, but more general
    
    Args:
        graph: Input graph (may have negative edge weights)
        source: Starting vertex index (0-based)
        goal: Target vertex index (0-based)
        
    Returns:
        Tuple of (distance, path, has_negative_cycle) where:
        - distance: Shortest path distance to goal (None if unreachable)
        - path: List of vertices on shortest path (empty if unreachable)
        - has_negative_cycle: True if negative cycle detected
        
    Note:
        If a negative cycle is detected, shortest path distances are undefined
        The returned path may not be meaningful in the presence of negative cycles
    """
    # STEP 1: INITIALIZATION
    # Set up data structures for the algorithm
    distances = [INFINITY] * graph.vertices  # Shortest distances from source
    predecessors = [None] * graph.vertices   # Predecessor array for path reconstruction
    distances[source] = 0.0                  # Source distance is 0
    
    # PHASE 1: RELAXATION PHASE (V-1 iterations)
    # The core insight: after i iterations, all shortest paths with ≤ i edges are correct
    # Since simple paths have at most V-1 edges, V-1 iterations suffice
    for iteration in range(graph.vertices - 1):
        # Flag to detect if any improvements were made in this iteration
        improved = False
        
        # Examine every edge in the graph for potential relaxation
        for u in range(graph.vertices):
            # Skip vertices that are not yet reachable
            if distances[u] == INFINITY:
                continue
                
            # Try to relax all outgoing edges from vertex u
            for edge in graph.adj[u]:
                new_distance = distances[u] + edge.weight
                
                # EDGE RELAXATION: Update if we found a better path
                if new_distance < distances[edge.to]:
                    distances[edge.to] = new_distance
                    predecessors[edge.to] = u
                    improved = True
        
        # EARLY TERMINATION OPTIMIZATION: If no improvements made, we're done
        # This can significantly speed up the algorithm on many graphs
        if not improved:
            break
    
    # PHASE 2: NEGATIVE CYCLE DETECTION
    # Perform one additional iteration to detect negative-weight cycles
    # If any distance can still be improved, a negative cycle exists
    has_negative_cycle = False
    for u in range(graph.vertices):
        if distances[u] == INFINITY:
            continue
            
        for edge in graph.adj[u]:
            # If we can still improve a distance, negative cycle detected
            if distances[u] + edge.weight < distances[edge.to]:
                has_negative_cycle = True
                break
        
        if has_negative_cycle:
            break

    # STEP 2: REACHABILITY CHECK
    if distances[goal] == INFINITY:
        return None, [], has_negative_cycle

    # STEP 3: PATH RECONSTRUCTION WITH CYCLE DETECTION
    # Reconstruct path while avoiding infinite loops due to negative cycles
    path = []
    curr = goal
    visited_path = set()  # Detect cycles in path reconstruction
    
    while curr is not None and curr not in visited_path:
        visited_path.add(curr)
        path.append(curr)
        
        if curr == source:  # Successfully reached source
            break
            
        curr = predecessors[curr]

    # Validate successful path reconstruction
    if not path or path[-1] != source:
        return None, [], has_negative_cycle
    
    # Return results: distance, path (reversed), and negative cycle flag
    return distances[goal], path[::-1], has_negative_cycle