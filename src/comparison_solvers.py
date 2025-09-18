import heapq
from typing import Optional, Tuple, List
from .graph import Graph

INFINITY = float('inf')

def dijkstra(graph: Graph, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
    """
    An implementation of Dijkstra's classic algorithm for finding the shortest
    path in a graph with non-negative edge weights.

    It works by iteratively selecting the unvisited vertex with the smallest known
    distance and relaxing its neighbors. A priority queue is used to efficiently
    retrieve the vertex with the minimum distance at each step.
    """
    distances = [INFINITY] * graph.vertices
    predecessors = [None] * graph.vertices
    distances[source] = 0.0
    
    # The priority queue stores tuples of (distance, vertex_id).
    # Python's heapq is a min-heap, so it always gives us the vertex with the smallest distance.
    pq = [(0.0, source)]
    
    while pq:
        dist, u = heapq.heappop(pq)

        # If we have already found a shorter path to 'u', we can ignore this one.
        if dist > distances[u]:
            continue
        
        # If we have reached the goal, we can stop early.
        if u == goal:
            break

        # For each neighbor 'v' of the current vertex 'u'...
        for edge in graph.adj[u]:
            # ...calculate the distance to 'v' through 'u'.
            new_dist = distances[u] + edge.weight
            # If this new path is shorter, update the distance and predecessor.
            if new_dist < distances[edge.to]:
                distances[edge.to] = new_dist
                predecessors[edge.to] = u
                heapq.heappush(pq, (new_dist, edge.to))

    # If the goal's distance is still infinity, it was not reachable.
    if distances[goal] == INFINITY:
        return None

    # Reconstruct the path by backtracking from the goal.
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        if curr == source: break
        curr = predecessors[curr]
    
    if not path or path[-1] != source: return None

    return distances[goal], path[::-1]


def bellman_ford(graph: Graph, source: int, goal: int) -> Tuple[Optional[float], List[int], bool]:
    """
    An implementation of the Bellman-Ford algorithm.
    It is slower than Dijkstra's but can handle graphs with negative edge weights.
    It works by relaxing all edges in the graph V-1 times. This repeated
    relaxation guarantees finding the shortest path if no negative cycles exist.

    Returns a tuple containing: (distance, path, has_negative_cycle).
    """
    distances = [INFINITY] * graph.vertices
    predecessors = [None] * graph.vertices
    distances[source] = 0.0

    # The core of the algorithm: relax every edge in the graph V-1 times.
    # After 'i' iterations, the algorithm has found all shortest paths of at most 'i' edges.
    for _ in range(graph.vertices - 1):
        for u in range(graph.vertices):
            if distances[u] == INFINITY: continue
            for edge in graph.adj[u]:
                if distances[u] + edge.weight < distances[edge.to]:
                    distances[edge.to] = distances[u] + edge.weight
                    predecessors[edge.to] = u
    
    # A final, V-th iteration is performed to detect negative-weight cycles.
    # If any path can still be shortened, it means there is a negative cycle.
    has_negative_cycle = False
    for u in range(graph.vertices):
        if distances[u] == INFINITY: continue
        for edge in graph.adj[u]:
            if distances[u] + edge.weight < distances[edge.to]:
                has_negative_cycle = True
                break
        if has_negative_cycle: break

    if distances[goal] == INFINITY:
        return None, [], has_negative_cycle

    # Reconstruct the path if the goal was reached.
    path = []
    curr = goal
    visited_path = set()
    while curr is not None and curr not in visited_path:
        visited_path.add(curr)
        path.append(curr)
        if curr == source: break
        curr = predecessors[curr]

    if not path or path[-1] != source: return None, [], has_negative_cycle
    
    return distances[goal], path[::-1], has_negative_cycle