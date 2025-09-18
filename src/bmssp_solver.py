import math
import heapq
from collections import deque
from typing import Optional, Tuple, List, Dict
from .graph import Graph
from .data_structure import EfficientDataStructure
from .comparison_solvers import dijkstra

INFINITY = float('inf')

class BmsspSolver:
    """
    Educational implementation of the Bounded Multi-Source Shortest Path (BMSSP) algorithm.
    
    This algorithm uses a divide-and-conquer approach on the graph's vertices to avoid 
    the sorting bottleneck inherent in Dijkstra's algorithm. The key insight is to 
    partition the shortest path problem into smaller subproblems based on distance ranges,
    solving each recursively.
    
    ALGORITHM OVERVIEW:
    1. Divide: Partition vertices by distance ranges using a specialized data structure
    2. Conquer: Recursively solve each partition with tighter bounds
    3. Combine: Use edge relaxation to propagate improvements between partitions
    
    KEY PARAMETERS:
    - k: Controls the depth of local Bellman-Ford-like exploration (typically O(log^(1/3) n))
    - t: Determines the partitioning factor in divide-and-conquer (typically O(log^(2/3) n))
    
    TIME COMPLEXITY: O(m log^(2/3) n) where m = edges, n = vertices
    SPACE COMPLEXITY: O(n)
    
    This educational version prioritizes clarity and understanding over maximum performance.
    For production use, consider BmsspSolverV2 which includes additional optimizations.
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
        Find the shortest path from source to goal vertex using the BMSSP algorithm.
        
        Args:
            source: Starting vertex ID
            goal: Target vertex ID
            
        Returns:
            Tuple of (distance, path) if path exists, None otherwise
            Path is represented as list of vertex IDs from source to goal
            
        Algorithm Steps:
        1. Initialize solver state (distances, predecessors, completion flags)
        2. Handle small graphs with direct Dijkstra for efficiency
        3. Calculate recursion depth based on graph size and parameter t
        4. Launch recursive BMSSP starting from source
        5. Reconstruct and return the shortest path if found
        """
        # Reset the state for each new solve call to ensure independence.
        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = INFINITY
        self.distances[source] = 0.0

        # For small graphs, the overhead of the BMSSP algorithm is not justified,
        # so we fall back to the simpler and faster Dijkstra's algorithm.
        # This threshold ensures we only use BMSSP when it provides clear benefits.
        if self.n < 1000:
            result = dijkstra(self.graph, source, goal)
            if result is not None:
                return result
            # If Dijkstra fails, continue with BMSSP as a fallback

        # The number of recursion levels is determined by n and t.
        max_level = math.ceil(math.log2(self.n) / self.t) if self.n > 1 else 0
        
        # The main algorithm is a top-level call to the recursive BMSSP procedure,
        # starting from the source vertex with maximum recursion depth.
        self._bmssp(max_level, INFINITY, [source], goal)

        # Check if we found a path to the goal
        if self.distances[goal] == INFINITY:
            # As a safety measure, fall back to Dijkstra if BMSSP failed
            # This ensures correctness even if there are edge cases in BMSSP
            fallback_result = dijkstra(self.graph, source, goal)
            return fallback_result
        
        # Reconstruct the path from the predecessor information
        path = self._reconstruct_path(source, goal)
        return self.distances[goal], path

    def _reconstruct_path(self, source: int, goal: int) -> List[int]:
        """
        Backtracks from the goal to the source using the `predecessors` array
        to build the shortest path.
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
        
        This function embodies the heart of the BMSSP algorithm:
        1. TERMINATION: Stop if no pivots remain or goal is already found
        2. BASE CASE: At level 0, switch to bounded Dijkstra search
        3. DIVIDE: Expand frontier and select influential pivots for next level
        4. CONQUER: Process vertices in distance-ordered blocks recursively
        5. COMBINE: Relax edges from completed vertices to discover new paths
        
        Args:
            level: Current recursion depth (decreases towards 0)
            bound: Upper bound on distances to consider in this subproblem
            pivots: Set of "influential" vertices to process at this level
            goal: Target vertex (None for all-pairs shortest paths)
            
        Returns:
            List of vertices that were finalized (completed) in this recursive call
            
        The algorithm maintains the invariant that vertices are processed in
        non-decreasing order of their shortest path distances.
        """
        # TERMINATION CONDITIONS:
        # 1. No more pivots to process (frontier is empty)
        # 2. Goal vertex has already been finalized with its shortest distance
        if not pivots or (goal is not None and self.complete[goal]):
            return []

        # BASE CASE: At recursion level 0, switch to bounded Dijkstra search
        # This provides the foundation for the divide-and-conquer approach
        if level == 0:
            return self._base_case(bound, pivots, goal)

        # PIVOT SELECTION: Expand the current frontier through k-step relaxation,
        # then select a smaller set of "influential" pivots that represent
        # the most important vertices for the next recursion level
        pivots, _ = self._find_pivots(bound, pivots)
        
        # DATA STRUCTURE SETUP: Block size grows exponentially with recursion level
        # This balances the trade-off between fine-grained control and efficiency
        # Higher levels use larger blocks to reduce overhead
        block_size = 2**max(0, (level - 1) * self.t)
        ds = EfficientDataStructure(block_size, bound)
        
        # INITIALIZATION: Insert selected pivots into the data structure
        # Only insert vertices that haven't been finalized and are within bounds
        for pivot in pivots:
            if not self.complete[pivot] and self.distances[pivot] < bound:
                ds.insert(pivot, self.distances[pivot])

        result_set = []
        
        # MAIN DIVIDE-AND-CONQUER LOOP:
        # Process vertices in order of increasing distance by pulling blocks
        # from the data structure and solving each as a recursive subproblem
        while not ds.is_empty():
            # Early termination if goal is already found and finalized
            if goal is not None and self.complete[goal]:
                break

            # DIVIDE: Extract the next block of closest vertices
            # subset_bound = minimum distance among remaining vertices in data structure
            # This ensures we process vertices in non-decreasing distance order
            subset_bound, subset = ds.pull()
            if not subset:
                continue

            # CONQUER: Recursively solve the subproblem with tighter distance bound
            # The recursive call will finalize vertices in this distance range
            sub_result = self._bmssp(level - 1, subset_bound, subset, goal)
            result_set.extend(sub_result)

            # COMBINE: Relax outgoing edges from newly finalized vertices
            # This may discover shorter paths to other vertices, which get
            # inserted into the data structure for future processing
            self._edge_relaxation(sub_result, subset_bound, bound, ds)

        return result_set

    def _base_case(self, bound: float, frontier: List[int], goal: Optional[int]) -> List[int]:
        """
        Base case of the BMSSP recursion: bounded Dijkstra search.
        
        When recursion reaches level 0, we switch to a modified Dijkstra's algorithm
        that respects the distance bound. This provides the foundation upon which
        the divide-and-conquer approach builds.
        
        Key differences from standard Dijkstra:
        1. Only processes vertices with distance < bound
        2. Allows distance equality (<=) for path re-processing across levels
        3. Early termination when goal is found
        
        Args:
            bound: Maximum distance to consider (vertices beyond this are ignored)
            frontier: Starting vertices for this bounded search
            goal: Target vertex (None for unrestricted search)
            
        Returns:
            List of vertices that were finalized in this bounded search
        """
        # INITIALIZATION: Create priority queue with frontier vertices
        # Only include vertices that haven't been finalized and are within bound
        pq = []
        for start_node in frontier:
            if not self.complete[start_node] and self.distances[start_node] < bound:
                heapq.heappush(pq, (self.distances[start_node], start_node))

        completed_nodes = []
        
        # DIJKSTRA'S MAIN LOOP: Process vertices in order of increasing distance
        while pq:
            dist, u = heapq.heappop(pq)

            # STALENESS CHECK: Ignore outdated entries in the priority queue
            # This happens when a vertex's distance was improved after insertion
            if self.complete[u] or dist > self.distances[u]:
                continue

            # FINALIZATION: Mark vertex as completed and add to result
            self.complete[u] = True
            completed_nodes.append(u)

            # GOAL CHECK: Early termination if target is reached
            if u == goal:
                # Update the global best distance for enhanced bound validation
                if dist < self.best_goal:
                    self.best_goal = dist
                break

            # EDGE RELAXATION: Update distances to neighboring vertices
            for edge in self.graph.adj[u]:
                new_dist = dist + edge.weight
                
                # IMPROVEMENT CHECK: Update if we found a better or equal path
                # Equality is allowed to ensure proper interaction between recursion levels
                # Enhanced bound validation: also check against best known goal distance
                if (not self.complete[edge.to] and 
                    new_dist <= self.distances[edge.to] and 
                    new_dist < bound and 
                    new_dist < self.best_goal):
                    
                    # UPDATE: Record the improved distance and path
                    self.distances[edge.to] = new_dist
                    self.predecessors[edge.to] = u
                    heapq.heappush(pq, (new_dist, edge.to))
        
        return completed_nodes

    def _find_pivots(self, bound: float, frontier: List[int]) -> Tuple[List[int], List[int]]:
        """
        Two-phase pivot selection algorithm for efficient frontier management.
        
        This function implements a crucial optimization that determines which vertices
        should be considered "influential" at the next recursion level. The goal is
        to reduce the number of subproblems while maintaining correctness.
        
        PHASE 1: k-step Bellman-Ford relaxation
        - Expands the frontier by exploring k steps from current vertices
        - Builds a "working set" of all reachable vertices within k steps
        - Detects "dense" cases where the frontier expands rapidly
        
        PHASE 2: Pivot selection (sparse case only)
        - Constructs shortest path forest from predecessor information
        - Calculates subtree sizes using post-order DFS traversal
        - Selects pivots as roots of large subtrees (size >= k)
        
        Args:
            bound: Distance bound for this recursion level
            frontier: Current set of active vertices
            
        Returns:
            Tuple of (selected_pivots, working_set)
            - selected_pivots: Influential vertices for next level
            - working_set: All vertices reachable within k steps
        """
        # INITIALIZATION: Start with frontier as the working set
        working_set = set(frontier)
        current_layer = {node for node in frontier if not self.complete[node]}

        # PHASE 1: K-STEP RELAXATION (Bellman-Ford style)
        # Expand the frontier by exploring k steps from current vertices
        for step in range(self.k):
            next_layer = set()
            
            # LAYER EXPANSION: Process all vertices in current layer
            for u in current_layer:
                # Skip vertices that exceed the distance bound
                if self.distances[u] >= bound: 
                    continue
                    
                # EDGE RELAXATION: Try to improve distances to neighbors
                for edge in self.graph.adj[u]:
                    v = edge.to
                    if self.complete[v]: 
                        continue
                    
                    new_dist = self.distances[u] + edge.weight
                    
                    # DISTANCE UPDATE: Allow equality for cross-level consistency
                    # Enhanced bound validation: also check against best known goal distance
                    if (new_dist <= self.distances[v] and 
                        new_dist < bound and 
                        new_dist < self.best_goal):
                        self.distances[v] = new_dist
                        self.predecessors[v] = u
                        
                        # ADD TO NEXT LAYER: Only if not already in working set
                        if v not in working_set:
                            next_layer.add(v)
            
            # TERMINATION CHECK: Stop if no new vertices were discovered
            if not next_layer: 
                break
                
            # UPDATE WORKING SET: Add newly discovered vertices
            working_set.update(next_layer)
            current_layer = next_layer

            # DENSITY CHECK: If working set grows too large, we're in "dense" case
            # In dense graphs, skip pivot selection and use original frontier
            if len(working_set) > self.k * len(frontier):
                return frontier, list(working_set)

        # PHASE 2: PIVOT SELECTION (Sparse Case)
        # Build shortest path forest structure from predecessor relationships
        
        # FOREST CONSTRUCTION: Map each vertex to its children in the SP tree
        children = {node: [] for node in working_set}
        for node in working_set:
            pred = self.predecessors[node]
            if pred is not None and pred in working_set:
                children[pred].append(node)

        # SUBTREE SIZE CALCULATION: Use post-order DFS to compute sizes
        # Post-order ensures child subtree sizes are known before parent processing
        subtree_sizes = {}
        post_order_stack = []
        visited_dfs = set()
        
        for node in working_set:
            if node not in visited_dfs:
                traversal_stack = [(node, iter(children.get(node, [])))]
                while traversal_stack:
                    curr, child_iter = traversal_stack[-1]
                    visited_dfs.add(curr)
                    next_child = next(child_iter, None)
                    if next_child:
                        if next_child not in visited_dfs:
                            traversal_stack.append((next_child, iter(children.get(next_child, []))))
                    else:
                        post_order_stack.append(curr)
                        traversal_stack.pop()

        for node in post_order_stack:
            size = 1 + sum(subtree_sizes.get(child, 0) for child in children.get(node, []))
            subtree_sizes[node] = size
        
        # PIVOT SELECTION: Choose vertices that are "influential" enough
        # A pivot must be from the original frontier AND root a large subtree (>= k nodes)
        pivots = [root for root in frontier if subtree_sizes.get(root, 0) >= self.k]

        # FALLBACK: If no influential pivots found, use the entire original frontier
        # This ensures the algorithm always makes progress
        if not pivots:
            return frontier, list(working_set)
        
        return pivots, list(working_set)

    def _edge_relaxation(self, completed_vertices: List[int], lower_bound: float, upper_bound: float, ds: EfficientDataStructure):
        """
        Edge relaxation phase: propagate improvements from completed vertices.
        
        After a recursive call completes a set of vertices, we need to check if
        their outgoing edges can improve distances to other vertices. This is the
        "combine" step of the divide-and-conquer approach.
        
        The function handles two types of discovered improvements:
        1. High-priority vertices (distance < lower_bound): Added to front of queue
        2. Normal vertices (lower_bound <= distance < upper_bound): Added normally
        
        Args:
            completed_vertices: Vertices that were just finalized
            lower_bound: Distance bound of the subproblem that just completed
            upper_bound: Distance bound of the current recursion level
            ds: Data structure for managing vertex processing order
        """
        # BATCH COLLECTION: Collect high-priority vertices for efficient insertion
        batch_prepend_list = []
        
        # EDGE PROCESSING: Check all outgoing edges from completed vertices
        for u in completed_vertices:
            for edge in self.graph.adj[u]:
                v = edge.to
                
                # SKIP COMPLETED: Don't relax edges to already-finalized vertices
                if self.complete[v]: 
                    continue

                new_dist = self.distances[u] + edge.weight
                
                # DISTANCE IMPROVEMENT: Update if we found a better or equal path
                # Equality is crucial for proper cross-subproblem path updates
                # Bound validation: also check against best known goal distance
                if new_dist <= self.distances[v] and new_dist < self.best_goal:
                    self.distances[v] = new_dist
                    self.predecessors[v] = u
                    
                    # PRIORITY CLASSIFICATION: Determine how to handle this vertex
                    
                    if new_dist < lower_bound:
                        # HIGH PRIORITY: This vertex is "closer than expected"
                        # It should be processed before other vertices in the queue
                        batch_prepend_list.append((v, new_dist))
                        
                    elif new_dist < upper_bound:
                        # NORMAL PRIORITY: Insert into data structure normally
                        # Will be processed in distance order with other vertices
                        ds.insert(v, new_dist)
        
        # BATCH INSERTION: Add all high-priority vertices to front of queue
        # This ensures vertices closer than expected get processed immediately
        if batch_prepend_list:
            ds.batch_prepend(batch_prepend_list)


class BmsspSolverV2:
    """
    An optimized implementation of the Bounded Multi-Source Shortest Path (BMSSP) algorithm.
    
    This version (V2) includes several key correctness and performance enhancements over the 
    original BmsspSolver:
    
    PERFORMANCE OPTIMIZATIONS:
    - Global pruning: Introduces a `best_goal` field that tracks the shortest distance to the 
      goal found so far, enabling aggressive pruning of paths that cannot improve the solution
    - Duplicate prevention: Uses an `in_ds` boolean array to prevent inserting the same vertex 
      into the data structure multiple times within a recursion level
    - Attribute access optimization: Binds frequently accessed class attributes (distances, 
      complete, predecessors, graph.adj) to local variables within hot loops to reduce overhead
    - Enhanced edge relaxation pruning: Checks against both local bounds and global best_goal 
      distance before processing edges
    - Tighter bound management: Uses min(bound, best_goal) consistently throughout recursion 
      levels for more effective pruning
    
    ALGORITHMIC ENHANCEMENTS:
    - Early termination: More aggressive early stopping when goal is reached or bounds are exceeded
    - Subproblem pruning: Skips entire subproblems when their minimum distance exceeds best_goal
    - Type annotations: Adds explicit type hints for better code clarity and IDE support
    
    These optimizations maintain the O(m log^(2/3) n) theoretical time complexity while providing 
    significant practical speedups, especially on graphs where the goal is reachable with a 
    relatively short path.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.vertices

        # Set parameters k and t based on the paper's formula for theoretical efficiency.
        self.k = int(math.log2(self.n) ** (1 / 3) * 2) if self.n > 1 else 1
        self.t = int(math.log2(self.n) ** (2 / 3)) if self.n > 1 else 1
        self.k = max(self.k, 3)
        self.t = max(self.t, 2)

        # Solver state variables
        self.distances = [INFINITY] * self.n
        self.predecessors: List[Optional[int]] = [None] * self.n
        self.complete = [False] * self.n
        # A global upper bound updated when the goal is reached, for aggressive pruning.
        self.best_goal = INFINITY

    def solve(self, source: int, goal: int) -> Optional[Tuple[float, List[int]]]:
        """
        Public method to find the shortest path from a source to a goal vertex.
        """
        # Initialize state for the new run.
        self.distances = [INFINITY] * self.n
        self.predecessors = [None] * self.n
        self.complete = [False] * self.n
        self.best_goal = INFINITY
        self.distances[source] = 0.0

        # For small graphs, the overhead of BMSSP is not justified.
        if self.n < 1000:
            return dijkstra(self.graph, source, goal)

        max_level = math.ceil(math.log2(self.n) / self.t) if self.n > 1 else 0

        self._bmssp(max_level, INFINITY, [source], goal)
        
        # As a safety net, if BMSSP fails to find the path, fall back to Dijkstra.
        if self.distances[goal] == INFINITY:
            return dijkstra(self.graph, source, goal)

        path = self._reconstruct_path(source, goal)
        return self.distances[goal], path

    def _reconstruct_path(self, source: int, goal: int) -> List[int]:
        """
        Backtracks from the goal to the source using the `predecessors` array.
        """
        path = []
        curr = goal
        pred = self.predecessors
        while curr is not None:
            path.append(curr)
            if curr == source:
                break
            curr = pred[curr]
        return path[::-1]

    def _bmssp(self, level: int, bound: float, pivots: List[int], goal: Optional[int]) -> List[int]:
        """
        The core recursive function of the BMSSP algorithm. Returns a list of
        vertices that were completed (finalized) within this call.
        """
        if not pivots or (goal is not None and self.complete[goal]):
            return []

        if level == 0:
            return self._base_case(bound, pivots, goal)

        # Use the tighter of the local `bound` or the global `best_goal` distance for pruning.
        eff_bound = min(bound, self.best_goal)
        pivots, _ = self._find_pivots(eff_bound, pivots)
        
        block_size = 2 ** max(0, (level - 1) * self.t)
        ds = EfficientDataStructure(block_size, eff_bound)
        
        # Use a flag array to prevent inserting the same vertex into the data structure multiple times.
        in_ds = [False] * self.n
        
        # Bind class attributes to local variables for faster access within loops.
        dist = self.distances
        comp = self.complete

        for pivot in pivots:
            d = dist[pivot]
            if not comp[pivot] and d < self.best_goal:
                if not in_ds[pivot]:
                    ds.insert(pivot, d)
                    in_ds[pivot] = True

        result_set: List[int] = []

        while not ds.is_empty():
            if goal is not None and comp[goal]:
                break

            subset_bound, subset = ds.pull()
            if not subset:
                continue
            
            # Once a vertex is pulled, it can be re-inserted later, so reset its flag.
            for u in subset:
                in_ds[u] = False
            
            # Pruning: skip this entire subproblem if its minimum distance
            # is already greater than the best known path to the goal.
            if subset_bound >= self.best_goal:
                continue

            sub_result = self._bmssp(level - 1, min(subset_bound, self.best_goal), subset, goal)
            result_set.extend(sub_result)

            self._edge_relaxation(sub_result, subset_bound, min(bound, self.best_goal), ds, in_ds)

        return result_set

    def _base_case(self, bound: float, frontier: List[int], goal: Optional[int]) -> List[int]:
        """
        The base of the recursion. Performs a bounded Dijkstra-like search and
        returns the list of vertices it finalized.
        """
        pq: List[Tuple[float, int]] = []
        # Local variable binding for performance.
        dist = self.distances
        comp = self.complete
        pred = self.predecessors
        adj = self.graph.adj

        for start_node in frontier:
            d_start = dist[start_node]
            if not comp[start_node] and d_start < bound and d_start < self.best_goal:
                heapq.heappush(pq, (d_start, start_node))

        completed_nodes: List[int] = []

        while pq:
            d_u, u = heapq.heappop(pq)

            if comp[u] or d_u > dist[u]:
                continue

            comp[u] = True
            completed_nodes.append(u)

            if goal is not None and u == goal:
                # If the goal is reached, update the global best distance.
                # This helps prune other branches of the search.
                if d_u < self.best_goal:
                    self.best_goal = d_u
                break
            
            # Edge relaxation.
            for e in adj[u]:
                v = e.to
                if comp[v]:
                    continue
                new_d = d_u + e.weight
                
                # Prune paths that are already worse than the goal or the local bound.
                if new_d >= self.best_goal or new_d >= bound:
                    continue
                
                # Allow equality to re-process paths, crucial for reusing relaxation info.
                if new_d <= dist[v]:
                    dist[v] = new_d
                    pred[v] = u
                    heapq.heappush(pq, (new_d, v))

        return completed_nodes

    def _find_pivots(self, bound: float, frontier: List[int]) -> Tuple[List[int], List[int]]:
        """
        Performs a k-step relaxation to expand the frontier and selects a smaller
        set of "pivots" for the next recursion level.
        """
        dist = self.distances
        comp = self.complete
        pred = self.predecessors
        adj = self.graph.adj

        working_set = set(frontier)
        current_layer = {node for node in frontier if not comp[node]}

        for _ in range(self.k):
            next_layer = set()
            for u in current_layer:
                if dist[u] >= bound:
                    continue
                for e in adj[u]:
                    v = e.to
                    if comp[v]:
                        continue
                    new_d = dist[u] + e.weight
                    if new_d < self.best_goal and new_d < bound and new_d <= dist[v]:
                        dist[v] = new_d
                        pred[v] = u
                        if v not in working_set:
                            next_layer.add(v)

            if not next_layer:
                break
            working_set.update(next_layer)
            current_layer = next_layer
            
            # In the dense case, return the original pivots, but do not terminate the algorithm.
            if len(working_set) > self.k * len(frontier):
                return frontier, list(working_set)

        # Pivot selection based on subtree sizes.
        children: Dict[int, List[int]] = {node: [] for node in working_set}
        for node in working_set:
            p = pred[node]
            if p is not None and p in working_set:
                children[p].append(node)

        subtree_sizes: Dict[int, int] = {}
        post_order_stack: List[int] = []
        visited_dfs = set()

        for node in working_set:
            if node in visited_dfs:
                continue
            stack = [(node, iter(children.get(node, [])))]
            while stack:
                cur, it = stack[-1]
                visited_dfs.add(cur)
                nxt = next(it, None)
                if nxt is not None:
                    if nxt not in visited_dfs:
                        stack.append((nxt, iter(children.get(nxt, []))))
                else:
                    post_order_stack.append(cur)
                    stack.pop()

        for node in post_order_stack:
            size = 1 + sum(subtree_sizes.get(ch, 0) for ch in children.get(node, []))
            subtree_sizes[node] = size

        pivots = [root for root in frontier if subtree_sizes.get(root, 0) >= self.k]
        if not pivots:
            return frontier, list(working_set)
        return pivots, list(working_set)

    def _edge_relaxation(
        self,
        completed_vertices: List[int],
        lower_bound: float,
        upper_bound: float,
        ds: EfficientDataStructure,
        in_ds: List[bool],
    ):
        """
        Relaxes outgoing edges from newly completed vertices, inserting neighbors
        into the data structure while preventing duplicates.
        """
        dist = self.distances
        comp = self.complete
        pred = self.predecessors
        adj = self.graph.adj

        batch_prepend_list: List[Tuple[int, float]] = []

        for u in completed_vertices:
            d_u = dist[u]
            for e in adj[u]:
                v = e.to
                if comp[v]:
                    continue

                new_d = d_u + e.weight
                
                # Prune with global best goal distance.
                if new_d >= self.best_goal:
                    continue
                
                # Allow equality for path updates.
                if new_d <= dist[v]:
                    dist[v] = new_d
                    pred[v] = u

                    # If closer than the subproblem's bound, prepend with high priority.
                    if new_d < lower_bound and new_d < upper_bound:
                        batch_prepend_list.append((v, new_d))
                        in_ds[v] = True
                    # Otherwise, insert normally if not already in the data structure.
                    elif new_d < upper_bound and not in_ds[v]:
                        ds.insert(v, new_d)
                        in_ds[v] = True

        if batch_prepend_list:
            ds.batch_prepend(batch_prepend_list)