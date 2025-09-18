from collections import deque

class EfficientDataStructure:
    """
    An implementation of the specialized data structure proposed in the paper.
    Its purpose is to manage the frontier of vertices to be explored without
    requiring a fully sorted priority queue, thus avoiding the O(log n) cost
    per vertex that characterizes Dijkstra's algorithm.

    It works by grouping vertices into "blocks". Instead of pulling one vertex
    at a time, it pulls an entire block of vertices that are likely close to
    each other in distance. This batch processing is key to its efficiency.
    """
    def __init__(self, block_size: int, bound: float):
        # Stores high-priority blocks of vertices that were relaxed with a distance
        # smaller than the current minimum. These must be processed first.
        self.batch_blocks = deque()
        # Stores standard blocks of vertices. New vertices are added here.
        self.sorted_blocks = []
        # The maximum number of vertices a single block can hold.
        self.block_size = block_size
        # A distance threshold; vertices with distances beyond this bound are ignored.
        self.bound = bound

    def insert(self, vertex: int, distance: float):
        """
        Inserts a vertex and its distance. If the current block is full, a new
        one is created. This amortizes the cost of block creation.
        """
        if distance < self.bound:
            if not self.sorted_blocks or len(self.sorted_blocks[-1]) >= self.block_size:
                self.sorted_blocks.append([])
            self.sorted_blocks[-1].append((vertex, distance))

    def batch_prepend(self, items: list[tuple[int, float]]):
        """
        Adds a block of vertices to the high-priority queue. This happens when
        a recursive call to `_bmssp` returns vertices with distances smaller
        than the current working minimum.
        """
        if items:
            self.batch_blocks.appendleft(list(items))

    def pull(self) -> tuple[float, list[int]]:
        """
        Extracts a block of vertices to be processed next. It prioritizes
        the `batch_blocks`. The extracted block is sorted internally before being
        returned, but this cost is less than maintaining a global sorted order.
        """
        block_to_process = None
        if self.batch_blocks:
            block_to_process = self.batch_blocks.popleft()
        elif self.sorted_blocks:
            # Heuristically select the block containing the overall minimum element.
            # This helps guide the search towards the most promising vertices.
            min_dist_in_blocks = [min(d for _, d in b) if b else float('inf') for b in self.sorted_blocks]
            min_block_idx = min(range(len(min_dist_in_blocks)), key=min_dist_in_blocks.__getitem__)
            block_to_process = self.sorted_blocks.pop(min_block_idx)

        if block_to_process:
            block_to_process.sort(key=lambda x: x[1])
            vertices = [v for v, d in block_to_process]
            min_dist = self.peek_min()
            return min_dist, vertices

        return self.bound, []

    def peek_min(self) -> float:
        """
        Returns the smallest distance currently in the structure without removing it.
        This is used to set the 'bound' for the next recursive call.
        """
        min_val = self.bound
        all_blocks = list(self.batch_blocks) + self.sorted_blocks
        for block in all_blocks:
            if block:
                min_val = min(min_val, min(d for v, d in block))
        return min_val

    def is_empty(self) -> bool:
        """Checks if there are any vertices left to process."""
        return not any(self.batch_blocks) and not any(self.sorted_blocks)