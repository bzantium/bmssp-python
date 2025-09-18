import math
import heapq
from collections import deque
from typing import Optional, Tuple, List

from .graph import Graph
from .data_structure import EfficientDataStructure
from .comparison_solvers import dijkstra

INFINITY = float('inf')

class BmsspSolver:
    """
    Educational implementation of the Bounded Multi-Source Shortest Path (BMSSP) algorithm.
    
    This algorithm uses a divide-and-conquer approach on graph vertices to avoid the sorting
    bottleneck inherent in Dijkstra's algorithm. The core idea is to divide the shortest path
    problem into smaller subproblems based on distance ranges and solve each recursively.
    
    Algorithm Overview:
    The BMSSP algorithm breaks the traditional O(m + n log n) barrier of Dijkstra's algorithm
    by achieving O(m log^(2/3) n) time complexity through three main phases:
    
    1. DIVIDE: Partition vertices by distance ranges using a specialized data structure
       - Uses EfficientDataStructure to organize vertices into distance-based buckets
       - Each bucket contains vertices within a specific distance range
       - This avoids the need to maintain a globally sorted priority queue
    
    2. CONQUER: Recursively solve each partition with tighter bounds
       - Apply the same algorithm recursively to each distance bucket
       - Use progressively tighter distance bounds to limit search scope
       - Employ pivot selection to reduce the frontier size at each level
    
    3. COMBINE: Propagate improvements between partitions via edge relaxation
       - After solving a partition, relax edges to neighboring partitions
       - Update distances in other buckets based on newly computed shortest paths
       - Insert newly discovered vertices into appropriate distance buckets
    
    Key Parameters:
    - k: Controls the depth of local Bellman-Ford-style exploration (typically O(log^(1/3) n))
         This parameter determines how many layers of neighbors we explore when finding pivots
    - t: Determines the branching factor in divide-and-conquer (typically O(log^(2/3) n))
         This controls how many distance buckets we create at each recursion level
    
    Theoretical Complexity:
    - Time: O(m log^(2/3) n) where m = number of edges, n = number of vertices
    - Space: O(n) for distance arrays and data structures
    
    This educational version prioritizes clarity and understanding over maximum performance.
    The implementation includes detailed comments explaining each step of the algorithm.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.vertices

        # Parameters 'k' and 't' are set to optimize the algorithm's theoretical
        # time complexity. 'k' controls the depth of the local Bellman-Ford-like
        # exploration, while 't' determines the partitioning factor in the
        # divide-and-conquer strategy.
        self.k = int(math.log2(self.n)**(1/3) * 2) if self.n > 1 else 1
        self.t = int(math.log2(self.n)**(2/3)) if self.n > 1 else 1
        self.k = max(self.k, 3)
        self.t = max(self.t, 2)

        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        # Track the best known distance to goal for enhanced bound validation
        self.best_goal = INFINITY

    def solve(self, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
        """
        Find the shortest path from source to goal using the BMSSP algorithm.
        
        This is the main entry point that orchestrates the entire BMSSP process:
        1. Initialize distance arrays and algorithm state
        2. Handle small graphs with Dijkstra (more efficient for small n)
        3. Compute recursion depth based on graph size
        4. Launch the recursive BMSSP divide-and-conquer process
        5. Reconstruct and return the shortest path if found
        
        Args:
            source: Starting vertex index (0-based)
            goal: Target vertex index (0-based)
            
        Returns:
            Tuple of (distance, path) if path exists, None otherwise
            Path is a list of vertex indices from source to goal
        """
        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = INFINITY
        self.distances[source] = 0.0

        # For small graphs, Dijkstra's algorithm is more efficient due to lower overhead
        # The BMSSP algorithm's complexity advantages only manifest on larger graphs
        if self.n < 1000:
            return dijkstra(self.graph, source, goal)

        max_level = math.ceil(math.log2(self.n) / self.t) if self.n > 1 else 0

        self._bmssp(max_level, INFINITY, [source], goal)

        if self.distances[goal] == INFINITY:
            return None

        path = self._reconstruct_path(source, goal)
        return self.distances[goal], path

    def _reconstruct_path(self, source: int, goal: int) -> List[int]:
        """
        Reconstruct the shortest path by backtracking from goal to source using predecessors.
        
        The algorithm maintains a predecessor array during the search process.
        Each vertex stores a reference to the previous vertex on its shortest path.
        We follow these references backward from the goal to build the complete path.
        
        Args:
            source: Starting vertex of the path
            goal: Ending vertex of the path
            
        Returns:
            List of vertex indices representing the shortest path from source to goal
        """
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            if curr == source:
                break
            curr = self.predecessors[curr]
        return path[::-1]

    def _bmssp(self, level: int, bound: float, pivots: List[int], goal: Optional[int]) -> List[int]:
        """
        Core recursive function implementing the divide-and-conquer BMSSP strategy.
        
        This function represents the heart of the BMSSP algorithm. It recursively applies
        the divide-and-conquer approach, with each recursive call handling a smaller
        subproblem with tighter distance bounds.
        
        Algorithm Steps:
        1. BASE CASE: If at maximum recursion depth, run bounded Dijkstra
        2. PIVOT SELECTION: Find strategic vertices that lie on many shortest paths
        3. PARTITIONING: Create distance-based buckets using EfficientDataStructure
        4. RECURSIVE CALLS: Process each bucket with the algorithm recursively
        5. EDGE RELAXATION: Propagate improvements between buckets
        
        Args:
            level: Current recursion depth (decreases toward 0)
            bound: Maximum distance to consider in this subproblem
            pivots: List of frontier vertices to expand from
            goal: Target vertex (None if solving general SSSP)
            
        Returns:
            List of vertices that were completed (finalized) in this call
        """
        # Early termination conditions:
        # 1. No more vertices to process
        # 2. Goal vertex has already been completed (shortest path found)
        if not pivots or (goal is not None and self.complete[goal]):
            return []

        # BASE CASE: At maximum recursion depth, switch to bounded Dijkstra
        # This provides the foundation for the recursive decomposition
        if level == 0:
            return self._base_case(bound, pivots, goal)

        # STEP 1: PIVOT SELECTION
        # Identify strategic vertices that are likely to lie on many shortest paths
        # This reduces the frontier size and focuses computation on important vertices
        pivots, _ = self._find_pivots(bound, pivots)

        # STEP 2: PARTITIONING SETUP
        # Calculate the bucket size for this recursion level
        # Larger buckets at higher levels, smaller buckets as we go deeper
        block_size = 2**max(0, (level - 1) * self.t)
        
        # Create the specialized data structure for distance-based partitioning
        # This structure maintains vertices organized by distance ranges
        ds = EfficientDataStructure(block_size, bound)

        # STEP 3: POPULATE DISTANCE BUCKETS
        # Insert all valid pivot vertices into the appropriate distance buckets
        # Only include vertices that haven't been completed and are within bounds
        for pivot in pivots:
            if not self.complete[pivot] and self.distances[pivot] < bound:
                ds.insert(pivot, self.distances[pivot])

        result_set = []

        # STEP 4: PROCESS EACH DISTANCE BUCKET RECURSIVELY
        # Continue until all buckets are processed or goal is found
        while not ds.is_empty():
            # Early termination: stop if we've already found the shortest path to goal
            if goal is not None and self.complete[goal]:
                break

            # Extract the next bucket of vertices with similar distances
            # subset_bound is the maximum distance allowed for this bucket
            subset_bound, subset = ds.pull()
            if not subset:
                continue

            # RECURSIVE CALL: Apply BMSSP to this smaller subproblem
            # Process vertices in this distance range with tighter bounds
            sub_result = self._bmssp(level - 1, subset_bound, subset, goal)
            result_set.extend(sub_result)

            # STEP 5: EDGE RELAXATION
            # Propagate improvements from completed vertices to other buckets
            # This may discover new vertices or improve existing distances
            self._edge_relaxation(sub_result, subset_bound, bound, ds)

        return result_set

    def _base_case(self, bound: float, frontier: List[int], goal: Optional[int]) -> List[int]:
        """
        Base case of BMSSP recursion: Perform bounded Dijkstra search.
        
        When we reach the deepest level of recursion (level = 0), we switch to a
        modified Dijkstra's algorithm that respects the distance bound. This provides
        the foundation that makes the recursive decomposition work.
        
        Key differences from standard Dijkstra:
        1. Only processes vertices with distance < bound
        2. Only explores edges that stay within the bound
        3. Maintains the global best_goal distance for pruning
        4. Works with a subset of vertices (frontier) rather than the entire graph
        
        Args:
            bound: Maximum distance to consider in this search
            frontier: Starting vertices for this bounded search
            goal: Target vertex (if any) for early termination
            
        Returns:
            List of vertices that were completed during this search
        """
        # Initialize priority queue with all valid frontier vertices
        # Only include vertices that haven't been completed and are within bounds
        pq = []
        for start_node in frontier:
            if not self.complete[start_node] and self.distances[start_node] < bound:
                heapq.heappush(pq, (self.distances[start_node], start_node))

        completed_nodes = []

        # Standard Dijkstra loop with distance bound enforcement
        while pq:
            dist, u = heapq.heappop(pq)

            # Skip if vertex already processed or distance is stale
            if self.complete[u] or dist > self.distances[u]:
                continue

            # Mark vertex as completed (shortest path found)
            self.complete[u] = True
            completed_nodes.append(u)

            # Early termination if we reached the goal
            if u == goal:
                if dist < self.best_goal:
                    self.best_goal = dist
                break

            # Relax all outgoing edges from the current vertex
            for edge in self.graph.adj[u]:
                new_dist = dist + edge.weight

                # Edge relaxation conditions:
                # 1. Target vertex not yet completed
                # 2. New distance is better than current known distance
                # 3. New distance respects the bound for this subproblem
                # 4. New distance is better than best known path to goal (pruning)
                if (not self.complete[edge.to] and
                    new_dist <= self.distances[edge.to] and
                    new_dist < bound and
                    new_dist < self.best_goal):

                    # Update shortest distance and predecessor
                    self.distances[edge.to] = new_dist
                    self.predecessors[edge.to] = u
                    
                    # Add to priority queue for future processing
                    heapq.heappush(pq, (new_dist, edge.to))

        return completed_nodes

    def _find_pivots(self, bound: float, frontier: List[int]) -> Tuple[List[int], List[int]]:
        """
        Two-phase pivot selection algorithm for efficient frontier management.
        
        This function implements a sophisticated strategy to reduce the frontier size
        by identifying "pivot" vertices that are likely to lie on many shortest paths.
        
        The algorithm works in two phases:
        
        PHASE 1: LOCAL EXPANSION (Bellman-Ford style)
        - Perform k rounds of edge relaxation starting from the frontier
        - This discovers vertices that are reachable within k hops
        - Builds a local shortest path tree rooted at frontier vertices
        
        PHASE 2: PIVOT IDENTIFICATION
        - Analyze the structure of the shortest path tree
        - Identify vertices with large subtrees (many descendants)
        - These vertices are likely to be on many shortest paths (good pivots)
        
        The pivot selection is crucial for algorithm efficiency:
        - Good pivots reduce the effective frontier size
        - This leads to smaller subproblems in recursive calls
        - Poor pivot selection can degrade to O(n log n) complexity
        
        Args:
            bound: Maximum distance to consider during expansion
            frontier: Current set of vertices to expand from
            
        Returns:
            Tuple of (pivots, working_set) where:
            - pivots: Selected strategic vertices for recursive processing
            - working_set: All discovered vertices during expansion
        """
        # Initialize data structures for the two-phase algorithm
        working_set = set(frontier)  # All vertices discovered so far
        current_layer = {node for node in frontier if not self.complete[node]}  # Active vertices

        # PHASE 1: LOCAL EXPANSION
        # Perform k rounds of Bellman-Ford-style relaxation
        # This builds a local shortest path tree within k hops of the frontier
        for _ in range(self.k):
            next_layer = set()

            # Process all vertices in the current expansion layer
            for u in current_layer:
                # Skip vertices that are too far away
                if self.distances[u] >= bound:
                    continue

                # Relax all outgoing edges from this vertex
                for edge in self.graph.adj[u]:
                    v = edge.to
                    if self.complete[v]:
                        continue

                    new_dist = self.distances[u] + edge.weight

                    # Standard edge relaxation with bound checking
                    if (new_dist <= self.distances[v] and
                        new_dist < bound and
                        new_dist < self.best_goal):
                        self.distances[v] = new_dist
                        self.predecessors[v] = u

                        # Add newly discovered vertices to next layer
                        if v not in working_set:
                            next_layer.add(v)

            # Stop if no new vertices were discovered
            if not next_layer:
                break

            # Update working set and prepare for next iteration
            working_set.update(next_layer)
            current_layer = next_layer

            # Safety check: if working set grows too large, fall back to original frontier
            # This prevents exponential blowup in pathological cases
            if len(working_set) > self.k * len(frontier):
                return frontier, list(working_set)

        # PHASE 2: PIVOT IDENTIFICATION
        # Analyze the structure of the shortest path tree to find good pivots
        
        # Build the tree structure: map each vertex to its children
        children = {node: [] for node in working_set}
        for node in working_set:
            pred = self.predecessors[node]
            if pred is not None and pred in working_set:
                children.setdefault(pred, []).append(node)

        # Calculate subtree sizes: vertices with large subtrees are good pivots
        # A large subtree indicates the vertex lies on many shortest paths
        subtree_sizes = {node: len(ch) for node, ch in children.items()}

        # Select pivots: frontier vertices with subtree size >= k
        # These are the most "influential" vertices in terms of shortest paths
        pivots = [root for root in frontier if subtree_sizes.get(root, 0) >= self.k]

        # Fallback: if no good pivots found, use the entire frontier
        # This ensures the algorithm still makes progress
        if not pivots:
            return frontier, list(working_set)

        return pivots, list(working_set)

    def _edge_relaxation(self, completed_vertices: List[int], lower_bound: float, upper_bound: float, ds: EfficientDataStructure):
        """
        Edge relaxation phase: Propagate improvements from completed vertices.
        
        After completing a set of vertices in a recursive call, we need to propagate
        the newly discovered shortest paths to other parts of the graph. This function
        performs this crucial "combine" step of the divide-and-conquer approach.
        
        The relaxation process:
        1. For each completed vertex, examine all outgoing edges
        2. Check if the new path through this vertex improves any distances
        3. Update distances and predecessors for improved paths
        4. Insert newly discovered/improved vertices into appropriate buckets
        
        Distance bucket management:
        - If new distance < lower_bound: Add to priority prepend list (higher priority)
        - If lower_bound <= new distance < upper_bound: Add to regular bucket
        - If new distance >= upper_bound: Ignore (will be handled in future calls)
        
        Args:
            completed_vertices: Vertices that were just completed in recursive call
            lower_bound: Minimum distance for current bucket
            upper_bound: Maximum distance for current recursion level
            ds: EfficientDataStructure managing distance buckets
        """
        # Collect vertices that need high-priority processing
        batch_prepend_list = []

        # Process all edges from completed vertices
        for u in completed_vertices:
            for edge in self.graph.adj[u]:
                v = edge.to

                # Skip vertices that are already completed
                if self.complete[v]:
                    continue

                # Calculate potential improvement through this edge
                new_dist = self.distances[u] + edge.weight

                # Check if this path improves the current best distance to v
                if new_dist <= self.distances[v] and new_dist < self.best_goal:
                    # Update the shortest distance and path
                    self.distances[v] = new_dist
                    self.predecessors[v] = u

                    # Categorize the vertex based on its new distance
                    if new_dist < lower_bound:
                        # High priority: distance is better than current bucket range
                        # These vertices should be processed before the current bucket
                        batch_prepend_list.append((v, new_dist))
                    elif new_dist < upper_bound:
                        # Normal priority: fits in current recursion level
                        # Insert into appropriate distance bucket
                        ds.insert(v, new_dist)
                    # Distances >= upper_bound are ignored (handled in parent recursion)

        # Batch insert high-priority vertices for efficient processing
        if batch_prepend_list:
            ds.batch_prepend(batch_prepend_list)
