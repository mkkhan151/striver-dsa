import sys
from collections import deque

sys.stdin = open("input.txt")
sys.stdout = open("output.txt", "w")


class Solution:
    def bfs(self, adj: list[list[int]]) -> list[int]:
        """Return the nodes of a graph traversed using breadth first search (BFS)"""
        visited = [False] * len(adj)
        q = deque()

        visited[0] = True
        q.append(0)
        output = []

        while q:
            node = q.popleft()
            output.append(node)
            for n in adj[node]:
                if not visited[n]:
                    visited[n] = True
                    q.append(n)
        return output

    def dfs(self, adj: list[list[int]]) -> list[int]:
        """
        Return the nodes of a graph traversed using deapth first search (DFS)
        """

        visited = [False] * len(adj)
        output: list[int] = []

        def traverse(node: int):
            visited[node] = True
            output.append(node)
            for n in adj[node]:
                if not visited[n]:
                    traverse(n)

        traverse(0)
        return output


if __name__ == "__main__":
    adj = [[1, 2], [0, 2], [0, 1, 3, 4], [2], [2]]
    sol = Solution()
    # output = sol.bfs(adj)
    output = sol.dfs(adj)
    print(output)
