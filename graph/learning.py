import sys

sys.stdin = open("input.txt")
sys.stdout = open("output.txt", "w")


if __name__ == "__main__":
    n, m = map(int, input().strip().split(" "))

    # Graph Adjacency Matrix Representation
    # An adjacency matrix of a graph is a two-dimensional array of size n x n, where n is the number of nodes in the graph, with the property that a[ i ][ j ] = 1 if the edge (vᵢ, vⱼ) is in the set of edges, and a[ i ][ j ] = 0 if there is no such edge. Space complexity is O(2xN)
    # adj_mat = [
    #     [0 for _ in range(n + 1)] for _ in range(n + 1)
    # ]  # an nxn matrix to represent vertices and edges

    # for _ in range(m):
    #     u, v = map(int, input().strip().split(" "))
    #     adj_mat[u][v] = 1 # assign weight in case of weighted graph
    #     adj_mat[v][u] = 1 # remove this in case of directed

    # for row in adj_mat:
    #     print(row)

    # Graph Adjacency List Representation
    # Adjacency list of graph is a one dimension array of size n, where n is the number of nodes in the graph, with property that a[i] = list of adjacent nodes to the ith node. Space complexity is O(2xE)
    adj_list = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().strip().split(" "))
        # u, v, wt = map(int, input().strip().split(" "))  # in case of weighted graph
        adj_list[u].append(v)
        adj_list[v].append(u)  # remove this in case of directed
        # for weighted graph
        # adj_list[u].append((v, wt))
        # adj_list[v].append((u, wt))  # remove this in case of directed

    for row in adj_list:
        print(row)
