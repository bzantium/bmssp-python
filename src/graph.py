class Edge:
    """
    Represents a directed, weighted edge in the graph.
    'to': The destination vertex of the edge.
    'weight': The cost associated with traversing the edge.
    """
    def __init__(self, to: int, weight: float):
        self.to = to
        self.weight = weight

    def __repr__(self):
        return f"Edge(to={self.to}, weight={self.weight})"

class Graph:
    """
    Represents a graph using an adjacency list. This structure is efficient
    for sparse graphs, where the number of edges is much smaller than the
    square of the number of vertices.
    'vertices': The total number of vertices in the graph.
    'adj': A list of lists, where adj[u] contains all outgoing edges from vertex u.
    """
    def __init__(self, vertices: int):
        if vertices < 0:
            raise ValueError("Number of vertices cannot be negative.")
        self.vertices = vertices
        self.adj = [[] for _ in range(vertices)]

    def add_edge(self, u: int, v: int, weight: float):
        """
        Adds a directed edge from vertex 'u' to vertex 'v' with a given 'weight'.
        """
        if not (0 <= u < self.vertices and 0 <= v < self.vertices):
            raise IndexError("Vertex index out of bounds.")
        self.adj[u].append(Edge(v, weight))

    def __repr__(self):
        edge_count = sum(len(e) for e in self.adj)
        return f"Graph(vertices={self.vertices}, edges={edge_count})"