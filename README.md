# Breaking the Sorting Barrier: A Python Implementation of the BMSSP Algorithm

## 🚀 Introduction

Welcome to this educational repository for the **Bounded Multi-Source Shortest Path (BMSSP)** algorithm. This project provides a clear and commented Python implementation of the groundbreaking shortest path algorithm introduced in the paper "[Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/abs/2504.17033)".

The primary goal of this repository is to serve as a learning tool for students, educators, and enthusiasts who want to understand the inner workings of modern graph algorithms. We'll explore not only the new BMSSP algorithm but also revisit classic algorithms like **Dijkstra's** and **Bellman-Ford** to fully appreciate the advancements made.

## 🧐 The Quest for the Shortest Path: A Brief History

Finding the shortest path between two points in a graph is one of the most fundamental problems in computer science. It has wide-ranging applications, from GPS navigation and network routing to logistics and bioinformatics.

### 1\. Dijkstra's Algorithm: The Greedy Gold Standard

For decades, **Dijkstra's algorithm** has been the go-to solution for the Single-Source Shortest Path (SSSP) problem on graphs with non-negative edge weights.

#### How it Works:

Dijkstra's algorithm works in a "greedy" fashion. It maintains a set of visited vertices and a priority queue of vertices to visit next.

1.  **Initialization**: Start with a source vertex `s`. Its distance is 0, and all other vertices have a distance of infinity.
2.  **Iteration**: At each step, extract the vertex `u` with the smallest distance from the priority queue.
3.  **Relaxation**: For each neighbor `v` of `u`, check if the path through `u` is shorter than the known distance to `v`. If `distance(u) + weight(u, v) < distance(v)`, update `distance(v)`.
4.  **Completion**: Repeat until all reachable vertices have been visited.

#### The "Sorting Barrier":

The efficiency of Dijkstra's algorithm hinges on the priority queue, which essentially sorts the vertices by their current shortest distance. With an advanced data structure like a Fibonacci heap, this leads to a time complexity of **$O(m + n \\log n)$**, where `n` is the number of vertices and `m` is the number of edges.

For a long time, this was considered the optimal solution. The `$n \log n$` term, known as the **"sorting barrier,"** seemed unbreakable because the algorithm fundamentally relies on repeatedly finding the "minimum" element.

### 2\. The Bellman-Ford Algorithm: Handling Negative Weights

What if some edges have negative weights? Dijkstra's greedy approach fails here because an initially longer path might eventually lead to a much shorter path via a negative edge.

#### How it Works:

The **Bellman-Ford algorithm** takes a more methodical, dynamic programming approach.

1.  **Initialization**: Similar to Dijkstra, set the source distance to 0 and all others to infinity.
2.  **Repeated Relaxation**: Relax *all* edges in the graph. Repeat this process `n-1` times.
3.  **Why n-1 times?**: A shortest path can have at most `n-1` edges. In each iteration `i`, the algorithm finds all shortest paths of length at most `i`.
4.  **Negative Cycle Detection**: After `n-1` iterations, perform one more pass. If any distance can still be improved, it means there is a negative-weight cycle in the graph.

The time complexity of Bellman-Ford is **$O(n \\cdot m)$**, which is slower than Dijkstra's but makes it more versatile for graphs with negative weights.

## ✨ The BMSSP Algorithm: Breaking the Barrier

The BMSSP algorithm challenges the long-held belief in Dijkstra's optimality. It introduces a novel approach that avoids the sorting bottleneck.

### Core Concepts:

#### 1\. Divide and Conquer

Instead of tackling the entire graph at once, BMSSP breaks the problem into smaller, more manageable subproblems. It processes vertices in distance-based "levels" or "buckets." By focusing only on a limited range of distances at a time, it reduces the scope of each search.

#### 2\. Frontier Reduction with Pivots

A key challenge in shortest path algorithms is managing the "frontier"—the set of vertices to explore next. If the frontier grows too large, the algorithm slows down.

BMSSP cleverly reduces the frontier by identifying **pivots**. Pivots are strategically chosen vertices that are likely to lie on many shortest paths. Instead of exploring the entire frontier, the algorithm focuses its efforts on expanding from these few, important pivots. This is achieved through a process similar to a limited-depth Bellman-Ford relaxation, which identifies influential nodes.

### The Result: A New Time Complexity

By combining these techniques, BMSSP bypasses the need to strictly sort all frontier vertices. It processes vertices in semi-sorted "batches" or "blocks," which is a much faster operation. This leads to its groundbreaking time complexity of **$O(m \\log^{2/3}n)$**, officially breaking the sorting barrier.

This represents a significant theoretical improvement over Dijkstra's **$O(m + n \\log n)$** complexity, especially on sparse graphs where **$m = O(n)$**, reducing the complexity from **$O(n \\log n)$** to **$O(n \\log^{2/3}n)$**.

## 🐍 Implementation in this Repository

This project provides Python code that implements these algorithms for both educational and practical purposes.

### 📁 Project Structure

  * `src/graph.py`: Defines the basic `Graph` and `Edge` classes for representing weighted directed graphs.
  * `src/bmssp_solver.py`: Contains **two implementations** of the BMSSP algorithm:
    - **`BmsspSolver`**: Educational version with extensive documentation and comments for learning
    - **`BmsspSolverV2`**: Optimized version with performance enhancements and aggressive pruning
  * `src/comparison_solvers.py`: Reference implementations of **Dijkstra's** and **Bellman-Ford's** algorithms for comparison.
  * `src/data_structure.py`: Specialized data structure used by the BMSSP algorithm for efficient vertex processing.
  * `src/graph_cache.py`: Intelligent caching system for fast dataset loading using pickle serialization.
  * `main.py`: Comprehensive benchmarking script that loads graphs, runs all algorithms, and compares their results.
  * `graph_loader.py`: Optimized functions to parse graph data from various formats (DIMACS, SNAP) with caching support.

### 🎓 Two Flavors of BMSSP

#### BmsspSolver (Educational Version)
Designed for learning and understanding:
- **Extensive documentation**: Every method and algorithm phase is thoroughly explained
- **Educational comments**: Step-by-step explanations of the divide-and-conquer approach
- **Stability features**: Includes fallback mechanisms to ensure correctness
- **Clear structure**: Prioritizes code clarity over maximum performance

#### BmsspSolverV2 (Optimized Version)
Designed for performance:
- **Global pruning**: Uses best-known goal distance for aggressive path pruning
- **Duplicate prevention**: Prevents redundant insertions into data structures
- **Micro-optimizations**: Local variable binding to reduce attribute access overhead
- **Enhanced edge relaxation**: More efficient pruning during edge processing

Both versions maintain the same **O(m log^(2/3) n)** theoretical time complexity while serving different purposes.

### 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd bmssp-python
    ```

2.  **Install dependencies:**
    This project uses `pandas` for efficient loading of large graphs.

    ```bash
    pip install pandas
    ```

3.  **Run the benchmarking script:**
    The script automatically downloads datasets if they don't exist locally and caches parsed graphs for faster subsequent runs.

#### 📊 Available Datasets

- **Rome99** (default): Road network of Rome, Italy (~3K vertices, ~9K edges)
- **LiveJournal**: Social network graph (~4M vertices, ~69M edges)
- **Pokec**: Slovak social network (~1.6M vertices, ~30M edges)  
- **California**: California road network (~2M vertices, ~5M edges)
- **Pennsylvania**: Pennsylvania road network (~1M vertices, ~3M edges)
- **Texas**: Texas road network (~1.4M vertices, ~3.8M edges)
- **Google**: Web graph from Google (~875K vertices, ~5M edges)
- **Stanford**: Stanford web graph (~281K vertices, ~2M edges)

#### 🎛️ Command Line Options

**Basic usage:**
```bash
# Run with default settings (Rome99 dataset, optimized solver)
python main.py

# Choose a specific dataset
python main.py --data livejournal
python main.py --data pokec
python main.py --data california
python main.py --data texas
python main.py --data google
python main.py --data stanford

# Choose BMSSP solver version
python main.py --solver v1  # Educational version
python main.py --solver v2  # Optimized version (default)

# Combine options
python main.py --data rome --solver v1
```

**Caching options:**
```bash
# Default: Use caching (load from cache if available, save new cache entries)
python main.py --data california

# Disable cache saving but still load from existing cache
python main.py --data california --no-cache

# Force reload from original data files, bypass cache completely
python main.py --data california --force-reload
```

**Get help:**
```bash
python main.py --help
```

#### 📈 What You'll See

The script will:
1. Download and prepare the selected dataset (if needed)
2. Load the graph from cache (if available) or parse from data files, then cache for future runs
3. Display basic graph statistics
4. Run the selected BMSSP version and measure execution time
5. Run Dijkstra's algorithm for comparison
6. Run Bellman-Ford algorithm (on smaller graphs only)
7. Compare results and execution times

**🚀 Performance Benefits of Caching:**
- **First run**: Parses data files and saves to cache (normal speed)
- **Subsequent runs**: Loads from cache in milliseconds (up to 100x faster for large datasets)
- **Automatic validation**: Cache is rebuilt if source files are modified

#### ⚡ Performance Tip

For significantly better performance on large graphs, consider running with **PyPy**:
```bash
pypy3 main.py --data livejournal --solver v2
```

## License

MIT

## References

  - **Primary Paper**: ["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"](https://arxiv.org/abs/2504.17033) by Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin (2025)
  - **Inspired by Rust Implementation**: [alphastrata/DunMaoSSSP](https://github.com/alphastrata/DunMaoSSSP.git)