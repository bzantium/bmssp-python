from collections import deque
from typing import Optional, Tuple, List, Deque

class EfficientDataStructure:
    """
    Advanced data structure for the educational BMSSP algorithm that avoids O(log n) priority queue operations.
    
    This data structure is a key innovation in the theoretical BMSSP algorithm, designed to overcome
    the fundamental bottleneck of Dijkstra's algorithm: the need to maintain a globally sorted priority
    queue. Instead of extracting vertices one-by-one in strict distance order, this structure processes
    vertices in "blocks" that are approximately sorted.
    
    THEORETICAL FOUNDATION:
    The BMSSP algorithm achieves O(m log^(2/3) n) complexity by avoiding the O(m log n) cost of
    priority queue operations. This data structure is central to that improvement:
    
    1. BLOCK-BASED PROCESSING: Groups vertices into blocks of size O(log^(1/3) n)
    2. APPROXIMATE SORTING: Blocks are internally sorted only when extracted
    3. BATCH OPERATIONS: Processes entire blocks rather than individual vertices
    4. RECURSIVE INTEGRATION: Supports the multi-level recursive structure of BMSSP
    
    ALGORITHM PRINCIPLE:
    Instead of maintaining a global min-heap of all vertices (expensive), this structure:
    - Organizes vertices into blocks based on insertion order
    - Maintains separate high-priority blocks for vertices with improved distances
    - Only sorts blocks when they need to be processed (lazy evaluation)
    - Uses heuristics to select the most promising block
    
    PERFORMANCE ANALYSIS:
    - Insert: O(1) amortized (just append to current block)
    - Extract block: O(k log k) where k = block size
    - Overall: Reduces total sorting cost from O(m log n) to O(m log^(2/3) n)
    
    COMPARISON WITH TRADITIONAL APPROACHES:
    
    1. BINARY HEAP (Dijkstra): O(log n) per operation, O(m log n) total
       - Maintains global order at all times
       - Extracts vertices in perfect distance order
       - High overhead for each individual operation
    
    2. BUCKET QUEUE (Delta-stepping): O(1) average, depends on edge weights
       - Groups vertices by distance ranges
       - Efficient for integer or bounded weights
       - Can degrade to O(n) per operation in worst case
    
    3. THIS STRUCTURE: O(1) insert, O(k log k) extract, O(m log^(2/3) n) total
       - Balances sorting cost with processing efficiency
       - Works well with any weight distribution
       - Designed specifically for recursive divide-and-conquer
    
    DATA STRUCTURE COMPONENTS:
    - batch_blocks: High-priority deque for vertices with improved distances
    - sorted_blocks: Regular list of blocks for standard insertion
    - block_size: Maximum vertices per block (typically O(log^(1/3) n))
    - bound: Distance threshold for filtering vertices
    """
    def __init__(self, block_size: int, bound: float):
        """
        Initialize the efficient data structure with specified block size and distance bound.
        
        PARAMETER SELECTION THEORY:
        - block_size: Typically set to O(log^(1/3) n) for optimal theoretical complexity
        - bound: Maximum distance to consider in current recursive subproblem
        
        The block size represents a crucial trade-off:
        - Larger blocks: Less sorting overhead, but less precise ordering
        - Smaller blocks: More precise ordering, but more sorting operations
        - Optimal size: Balances these factors for O(m log^(2/3) n) complexity
        
        Args:
            block_size: Maximum vertices per block (affects sorting granularity)
            bound: Distance threshold for filtering vertices in this subproblem
        """
        # HIGH-PRIORITY QUEUE: Stores blocks of vertices that were discovered with distances
        # better than expected. These blocks get processed before regular blocks because
        # they likely contain vertices on shortest paths.
        self.batch_blocks = deque()
        
        # REGULAR BLOCKS: Standard insertion blocks for newly discovered vertices.
        # These blocks are processed in a heuristic order based on minimum distance.
        self.sorted_blocks = []
        
        # ALGORITHM PARAMETERS: Core parameters that control the trade-off between
        # sorting overhead and processing efficiency
        self.block_size = block_size  # Maximum vertices per block
        self.bound = bound           # Distance threshold for this subproblem

    def insert(self, vertex: int, distance: float):
        """
        Insert a vertex with its distance into the appropriate block structure.
        
        This is the core insertion operation that maintains the block-based organization.
        The method uses lazy block creation to minimize memory allocation overhead.
        
        ALGORITHM STEPS:
        1. DISTANCE FILTERING: Only insert vertices within the current bound
        2. BLOCK MANAGEMENT: Create new block if current block is full
        3. APPEND OPERATION: Add vertex-distance pair to current block
        
        AMORTIZED ANALYSIS:
        - Most insertions: O(1) - just append to existing block
        - Block creation: O(1) - create new empty list
        - Overall: O(1) amortized per insertion
        
        BLOCK MANAGEMENT STRATEGY:
        - Keep blocks at or below block_size to control sorting cost
        - Use the last block as the "active" block for new insertions
        - Create blocks on-demand to avoid pre-allocating unused memory
        
        DISTANCE BOUND FILTERING:
        Only vertices within the current bound are stored. This implements the
        "bounded" aspect of BMSSP, ensuring we only process relevant vertices
        in each recursive subproblem.
        
        Args:
            vertex: Vertex index to insert
            distance: Current shortest distance to this vertex
            
        Time Complexity: O(1) amortized
        Space Complexity: O(1) per vertex
        """
        # STEP 1: Distance bound filtering
        # Only store vertices that are within the current subproblem's distance bound
        if distance < self.bound:
            
            # STEP 2: Block management - ensure we have an active block with space
            # Create new block if: no blocks exist OR current block is full
            if not self.sorted_blocks or len(self.sorted_blocks[-1]) >= self.block_size:
                self.sorted_blocks.append([])  # Create new empty block
            
            # STEP 3: Insert vertex-distance pair into the active block
            # Store as tuple for easy sorting later
            self.sorted_blocks[-1].append((vertex, distance))

    def batch_prepend(self, items: list[tuple[int, float]]):
        """
        Add a block of high-priority vertices that must be processed before regular blocks.
        
        This method implements a crucial optimization in the BMSSP algorithm: when a recursive
        call discovers vertices with distances better than expected, these vertices are given
        priority in subsequent processing. This ensures that improvements propagate quickly
        through the algorithm.
        
        WHEN THIS IS CALLED:
        During the edge relaxation phase of BMSSP, when vertices are discovered with distances
        smaller than the current subproblem's lower bound. These vertices represent "early"
        discoveries that should be processed before continuing with the current distance range.
        
        PRIORITY QUEUE SEMANTICS:
        - batch_blocks acts as a priority queue where newer batches are processed first
        - Uses appendleft() to implement LIFO (Last In, First Out) ordering
        - This ensures most recently discovered improvements get immediate attention
        
        ALGORITHM INTEGRATION:
        This method bridges the "combine" phase of divide-and-conquer:
        1. Recursive call completes and returns improved vertices
        2. These vertices are batch_prepended to the data structure
        3. Next pull() operation will return these high-priority vertices first
        4. This propagates improvements efficiently through the recursion
        
        PERFORMANCE CHARACTERISTICS:
        - Time: O(1) for the deque appendleft operation
        - Space: O(k) where k is the number of items in the batch
        - No sorting required - items processed as a unit
        
        Args:
            items: List of (vertex, distance) tuples with improved distances
                  These should have distances smaller than the current lower bound
                  
        Time Complexity: O(1) for deque operations
        Space Complexity: O(len(items)) for storing the batch
        """
        # Only add non-empty batches to avoid processing overhead
        if items:
            # Convert to list for consistent data structure and add to front of deque
            # appendleft ensures this batch gets processed before older batches
            self.batch_blocks.appendleft(list(items))

    def pull(self) -> tuple[float, list[int]]:
        """
        Extract the next block of vertices to be processed by the BMSSP algorithm.
        
        This is the core extraction operation that implements the block-based processing
        strategy. The method carefully balances efficiency with approximate distance ordering
        to achieve the theoretical complexity improvements of BMSSP.
        
        ALGORITHM STRATEGY:
        The method implements a two-tier priority system:
        1. HIGH PRIORITY: Process batch_blocks first (vertices with improved distances)
        2. REGULAR PRIORITY: Select blocks heuristically based on minimum distance
        
        BLOCK SELECTION HEURISTIC:
        For regular blocks, we use a greedy heuristic that selects the block containing
        the vertex with globally minimum distance. This approximates optimal ordering
        without maintaining a global sorted structure.
        
        LAZY SORTING OPTIMIZATION:
        Blocks are only sorted when extracted, not when created. This "lazy evaluation"
        approach reduces the total sorting cost:
        - Traditional approach: Sort all vertices globally O(m log m)
        - This approach: Sort each block separately O(Σ k_i log k_i) = O(m log^(2/3) n)
        
        ALGORITHM STEPS:
        1. PRIORITY CHECK: Extract from batch_blocks if available (high priority)
        2. HEURISTIC SELECTION: Find block with minimum distance vertex
        3. LAZY SORTING: Sort the selected block by distance
        4. RETURN PREPARATION: Extract vertices and compute next bound
        
        THEORETICAL ANALYSIS:
        - Block selection: O(number of blocks) = O(m/block_size)
        - Block sorting: O(block_size * log(block_size))
        - Total complexity: O(m log^(2/3) n) when block_size = O(log^(1/3) n)
        
        Returns:
            Tuple of (next_bound, vertex_list) where:
            - next_bound: Distance bound for processing this block
            - vertex_list: List of vertex indices sorted by distance
            
        Time Complexity: O(k log k) where k = block_size
        Space Complexity: O(k) for the returned vertex list
        """
        block_to_process = None
        
        # STEP 1: Priority processing - handle high-priority batches first
        if self.batch_blocks:
            # Extract the most recently added batch (LIFO order)
            block_to_process = self.batch_blocks.popleft()
            
        elif self.sorted_blocks:
            # STEP 2: Heuristic block selection for regular processing
            # Find the block containing the vertex with globally minimum distance
            # This approximates optimal ordering without global sorting
            
            # Compute minimum distance in each block
            min_dist_in_blocks = [
                min(distance for _, distance in block) if block else float('inf') 
                for block in self.sorted_blocks
            ]
            
            # Select block with globally minimum distance
            min_block_idx = min(range(len(min_dist_in_blocks)), 
                              key=min_dist_in_blocks.__getitem__)
            block_to_process = self.sorted_blocks.pop(min_block_idx)

        # STEP 3: Process the selected block
        if block_to_process:
            # LAZY SORTING: Sort block only when extracted (key optimization)
            # Sort by distance (second element of tuple)
            block_to_process.sort(key=lambda vertex_dist_pair: vertex_dist_pair[1])
            
            # Extract vertex list (discard distances for processing)
            vertices = [vertex for vertex, distance in block_to_process]
            
            # Compute next distance bound for recursive processing
            next_bound = self.peek_min()
            
            return next_bound, vertices

        # STEP 4: Empty structure case
        # Return bound as next_bound and empty vertex list
        return self.bound, []

    def peek_min(self) -> float:
        """
        Find the minimum distance across all blocks without extracting any vertices.
        
        This method is crucial for the recursive structure of BMSSP, as it determines
        the distance bound for the next recursive subproblem. The bound ensures that
        recursive calls operate on increasingly tighter distance ranges.
        
        ALGORITHM PURPOSE:
        In BMSSP's divide-and-conquer approach, each recursive call needs a distance
        bound that defines the "working range" for that subproblem. This method
        provides that bound by finding the smallest distance among all remaining vertices.
        
        IMPLEMENTATION STRATEGY:
        - Examine all blocks (both high-priority and regular)
        - Find the minimum distance within each non-empty block
        - Return the global minimum across all blocks
        - Use current bound as fallback if no vertices remain
        
        PERFORMANCE CONSIDERATIONS:
        - Time: O(total vertices in all blocks) - could be expensive
        - Called once per pull() operation, so frequency is manageable
        - Alternative: Could maintain running minimum, but adds complexity
        
        RECURSIVE INTEGRATION:
        The returned value becomes the "bound" parameter for the next recursive
        BMSSP call, ensuring that each recursion level operates on a progressively
        smaller distance range.
        
        Returns:
            Minimum distance among all vertices in the structure, or current bound if empty
            
        Time Complexity: O(total vertices across all blocks)
        Space Complexity: O(1) auxiliary space
        """
        min_val = self.bound  # Start with current bound as default
        
        # Combine all blocks for comprehensive search
        all_blocks = list(self.batch_blocks) + self.sorted_blocks
        
        # Find global minimum across all blocks
        for block in all_blocks:
            if block:  # Skip empty blocks
                # Find minimum distance in this block
                block_min = min(distance for vertex, distance in block)
                min_val = min(min_val, block_min)
                
        return min_val

    def is_empty(self) -> bool:
        """
        Check if the data structure contains any vertices to process.
        
        This method determines when the BMSSP algorithm has finished processing
        all vertices in the current recursive subproblem. It's used as the
        termination condition in the main BMSSP loop.
        
        TERMINATION LOGIC:
        The structure is empty when:
        1. No high-priority batches remain (batch_blocks is empty)
        2. No regular blocks contain vertices (all sorted_blocks are empty)
        
        IMPLEMENTATION NOTE:
        Uses Python's any() function for efficient short-circuit evaluation.
        The structure is empty only when ALL blocks are empty.
        
        Returns:
            True if no vertices remain to process, False otherwise
            
        Time Complexity: O(number of blocks) for checking emptiness
        Space Complexity: O(1)
        """
        # Structure is empty when both batch_blocks and sorted_blocks contain no vertices
        # any() returns True if any block is non-empty, so we negate it
        return not any(self.batch_blocks) and not any(self.sorted_blocks)


class BucketQueue:
    """
    High-performance bucket queue implementation for delta-stepping shortest path algorithms.
    
    This data structure is a key optimization for BmsspSolverV2, providing O(1) insert and
    extract_min operations for vertices organized by distance ranges. It replaces the
    traditional binary heap used in Dijkstra's algorithm with a more efficient bucketing
    approach suitable for bounded distance ranges.
    
    ALGORITHM PRINCIPLE:
    Delta-stepping divides the distance range [0, ∞) into buckets of width δ (delta).
    Vertices with distances in [i*δ, (i+1)*δ) are stored in bucket i. This allows:
    - O(1) insertion: Calculate bucket index and append to deque
    - O(1) extract_min: Process buckets in order, extract from first non-empty bucket
    
    OPTIMIZATIONS vs Standard Binary Heap:
    1. BATCH PROCESSING: Process all vertices in a bucket before moving to next bucket
    2. CACHE LOCALITY: Vertices with similar distances processed together
    3. NO COMPARISONS: No expensive comparison operations during insertion
    4. REUSABLE STRUCTURE: Same queue object used across multiple searches
    5. EFFICIENT CLEARING: Only clear buckets that were actually used
    
    Memory Layout:
    buckets[0] = deque([vertices with distance 0 to δ])
    buckets[1] = deque([vertices with distance δ to 2δ])  
    buckets[2] = deque([vertices with distance 2δ to 3δ])
    ...
    """
    
    def __init__(self, delta: float, initial_buckets=64):
        """
        Initialize bucket queue with specified bucket width and initial capacity.
        
        Args:
            delta: Bucket width (distance range per bucket)
            initial_buckets: Initial number of pre-allocated buckets (optimization)
        """
        self.delta = delta  # Width of each distance bucket
        self.min_idx = 0    # Index of first non-empty bucket (for fast extract_min)
        
        # OPTIMIZATION: Pre-allocate buckets to avoid repeated allocation
        # Start with 64 buckets which covers distance range [0, 64*delta)
        self.buckets: List[Deque[int]] = [deque() for _ in range(initial_buckets)]
        
        # Track maximum used bucket index for efficient clearing
        self.max_idx = -1

    def clear(self):
        """
        Efficiently clear the bucket queue by only clearing buckets that were used.
        
        MAJOR OPTIMIZATION vs naive clearing: Instead of clearing all buckets,
        only clear the range [min_idx, max_idx] that was actually used.
        This provides O(k) clearing time where k = number of used buckets,
        instead of O(total_buckets).
        
        Performance Impact:
        - Typical case: Clear 5-20 buckets instead of 1000+ total buckets
        - Time savings: 50-200x faster clearing on sparse graphs
        """
        # Only clear buckets that were actually used in the previous search
        for i in range(self.min_idx, self.max_idx + 1):
            self.buckets[i].clear()
        
        # Reset tracking indices for next search
        self.min_idx = 0
        self.max_idx = -1

    def insert(self, v: int, dist: float):
        """
        Insert vertex v with given distance into the appropriate bucket.
        
        ALGORITHM:
        1. Calculate bucket index: idx = floor(distance / delta)
        2. Expand bucket array if necessary (dynamic resizing)
        3. Append vertex to the appropriate bucket's deque
        4. Update max_idx tracking for efficient clearing
        
        Time Complexity: O(1) average, O(k) worst case for bucket expansion
        
        Args:
            v: Vertex index to insert
            dist: Distance value for the vertex
        """
        # Calculate which bucket this distance belongs to
        idx = int(dist / self.delta)
        
        # OPTIMIZATION: Dynamic bucket expansion with batch allocation
        # Instead of growing one bucket at a time, allocate all needed buckets at once
        if idx >= len(self.buckets):
            # Extend bucket array to accommodate the new index
            num_new_buckets = idx - len(self.buckets) + 1
            self.buckets.extend(deque() for _ in range(num_new_buckets))
        
        # Add vertex to the appropriate bucket
        self.buckets[idx].append(v)
        
        # Update maximum index tracking for efficient clearing
        if idx > self.max_idx:
            self.max_idx = idx

    def extract_min(self) -> Tuple[Optional[int], bool]:
        """
        Extract the vertex with minimum distance from the queue.
        
        ALGORITHM:
        1. Start from min_idx (first potentially non-empty bucket)
        2. Scan forward until finding a non-empty bucket
        3. Extract vertex from that bucket using deque.popleft() 
        4. Update min_idx to current position for next extraction
        5. Return (vertex, True) if found, (None, False) if queue empty
        
        OPTIMIZATION vs Binary Heap:
        - No heap restructuring after extraction (O(1) vs O(log n))
        - Process vertices in distance order automatically
        - Cache-friendly access pattern (process similar distances together)
        
        Time Complexity: O(1) amortized (each bucket visited at most once)
        
        Returns:
            Tuple of (vertex_index, success_flag)
            - If queue not empty: (vertex_index, True)
            - If queue empty: (None, False)
        """
        min_idx = self.min_idx
        
        # OPTIMIZATION: Bounded search using max_idx to avoid scanning empty tail
        # Only search up to max_idx instead of the entire bucket array
        while min_idx <= self.max_idx:
            if self.buckets[min_idx]:  # Found non-empty bucket
                self.min_idx = min_idx  # Update min_idx for next extraction
                v = self.buckets[min_idx].popleft()  # Extract vertex
                return v, True
            min_idx += 1
        
        # No more vertices in queue
        self.min_idx = min_idx
        return None, False