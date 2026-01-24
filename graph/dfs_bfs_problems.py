import sys
from collections import deque
from typing import List, Set

sys.stdin = open("input.txt")
sys.stdout = open("output.txt", "w")


class Solution:
    # Find Number of Provinces
    def findCircleNum(self, isConnected: list[list[int]]) -> int:
        """Returns the total number of provinces where a province is the group of connected cities(vertices) directly or indirectly."""
        visited = [False] * len(isConnected)

        def dfs(city: int):
            visited[city] = True
            for v in range(len(isConnected)):
                if isConnected[city][v] == 1 and not visited[v]:
                    dfs(v)

        provinces = 0
        for city in range(len(isConnected)):
            if not visited[city]:
                provinces += 1
                dfs(city)

        return provinces

    # Count connected components
    def countComponents(self, V: int, edges: list[list[int]]) -> int:
        """Return the total connected components in the graph"""

        # Create adj list from edge list
        adj_list = [[]] * V
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        visited = [False] * V

        components = 0

        # Traverse all nodes in the graph
        for node in range(V):
            # if the node is not visited, it's a new component
            if not visited[node]:
                components += 1

                # Start BFS from this node
                q = deque()
                q.append(node)
                visited[node] = True

                # Perform BFS
                while q:
                    n = q.popleft()

                    # visit all unvisited nodes
                    for nbr in adj_list[n]:
                        if not visited[nbr]:
                            visited[nbr] = True
                            q.append(nbr)
        return components

    def orangesRotting(self, grid: list[list[int]]) -> int:
        """Return the number of minutes to rotten all oranges"""
        if len(grid) == 0:
            return 0

        rotten = deque()
        total_oranges = 0
        count = 0
        total_time = 0

        m, n = len(grid), len(grid[0])
        # Traverse the grid to count and collect initial rotten oranges
        for i in range(m):
            for j in range(n):
                # count any fresh or rotten orange
                if grid[i][j] != 0:
                    total_oranges += 1
                # add rotten orange to queue
                if grid[i][j] == 2:
                    rotten.append((i, j))

        # direction vectors for four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # BFS traversal of grid
        while rotten:
            # Number of oranges to process this minute
            k = len(rotten)
            count += k

            # Process all rotten oranges at this level
            for _ in range(k):
                x, y = rotten.popleft()
                # check all four directions
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    # skip if out of bound or not a fresh orange
                    if nx < 0 or ny < 0 or nx >= m or ny >= n or grid[nx][ny] != 1:
                        continue

                    # mark orange as rotten
                    grid[nx][ny] = 2

                    # add to queue for next round
                    rotten.append((nx, ny))

            # if queue still has items, increment minutes
            if rotten:
                total_time += 1

        # return minutes if all rotted, otherwise -1
        return total_time if count == total_oranges else -1

    def floodFill(
        self, image: list[list[int]], sr: int, sc: int, color: int
    ) -> list[list[int]]:
        """flood fill the image from the starting point to all the connected points of same color as starting point"""
        if image[sr][sc] == color:
            return image
        orig_color = image[sr][sc]
        m, n = len(image), len(image[0])

        # direction vector for four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Using DFS
        # def dfs_fill(r: int, c: int):
        #     image[r][c] = color
        #     for dr, dc in directions:
        #         nr, nc = r + dr, c + dc
        #         # if pixel out of bound or not same as the origin color, skip it
        #         if (
        #             nr < 0
        #             or nc < 0
        #             or nr >= m
        #             or nc >= n
        #             or image[nr][nc] != orig_color
        #         ):
        #             continue
        #         dfs_fill(nr, nc)

        # dfs_fill(sr, sc)

        # using BFS
        filled = deque()
        filled.append((sr, sc))
        while filled:
            r, c = filled.popleft()

            # paint the pixel as given color
            image[r][c] = color

            # Check for four directions
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # if pixel out of bound or not same as the origin color, skip it
                if (
                    nr < 0
                    or nc < 0
                    or nr >= m
                    or nc >= n
                    or image[nr][nc] != orig_color
                ):
                    continue
                # add to queue
                filled.append((nr, nc))

        return image

    def isCycle(self, V: int, adj: list[list[int]]) -> bool:
        """Return True if there is cycle in the graph otherwise false"""
        visited = [False] * V

        for i in range(V):
            if not visited[i]:
                if self.__detect(i, adj, visited):
                    return True
        return False

    def __detect(self, src: int, adj: list[list[int]], visited: list[bool]) -> bool:
        visited[src] = True
        q = deque()

        q.append((src, -1))
        while q:
            node, parent = q.popleft()
            for adj_node in adj[node]:
                if not visited[adj_node]:
                    visited[adj_node] = True
                    q.append((adj_node, node))
                elif parent != adj_node:
                    return True
        return False

    def updateMatrix(self, mat: list[list[int]]) -> list[list[int]]:
        """Find the nearest 0 for each cell."""
        m, n = len(mat), len(mat[0])
        q = deque()  # for BFS
        visited = [[False for _ in range(n)] for _ in range(m)]
        ans = [[-1 for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    visited[i][j] = True
                    q.append((i, j, 0))

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while q:
            i, j, distance = q.popleft()
            ans[i][j] = distance
            # check all four directions for neighbors
            for di, dj in directions:
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= m or nj < 0 or nj >= n or visited[ni][nj]:
                    # skip if visited or out of bound
                    continue
                # mark as visited
                visited[ni][nj] = True
                # add to queue with distance + 1
                q.append((ni, nj, distance + 1))
        return ans

    def solve(self, board: list[list[str]]) -> None:
        """
        capture regions that are surrounded by "X"
        """
        rows, cols = len(board), len(board[0])
        # handle empty matrix
        if rows == 0 or cols == 0:
            return
        visited = [[False] * cols for _ in range(rows)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def dfs(r: int, c: int):
            visited[r][c] = True
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    nr < 0
                    or nr >= rows
                    or nc < 0
                    or nc >= cols
                    or board[nr][nc] == "X"
                    or visited[nr][nc]
                ):
                    # skip if out of bound or visited or cell is X
                    continue
                dfs(nr, nc)

        # Check the edges of board and mark them as visited if "O"
        # DFS will mark all the connected node as visited
        for row in range(rows):
            if not visited[row][0] and board[row][0] == "O":
                dfs(row, 0)
            if not visited[row][cols - 1] and board[row][cols - 1] == "O":
                dfs(row, cols - 1)

        for col in range(cols):
            if not visited[0][col] and board[0][col] == "O":
                dfs(0, col)
            if not visited[rows - 1][col] and board[rows - 1][col] == "O":
                dfs(rows - 1, col)

        # Now mark all the ramaining unvisited "O" as "X"
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == "O" and not visited[i][j]:
                    board[i][j] = "X"

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands.
        """
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        islands = 0

        def dfs(r, c):
            visited[r][c] = True

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    nr < 0
                    or nc < 0
                    or nr >= rows
                    or nc >= cols
                    or visited[nr][nc]
                    or grid[nr][nc] == "0"
                ):
                    continue
                dfs(nr, nc)

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != "0" and not visited[row][col]:
                    islands += 1
                    dfs(row, col)
        return islands

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []

        rows, cols = len(heights), len(heights[0])
        pacific_reachable = set()
        atlantic_reachable = set()

        def dfs(r: int, c: int, reachable: Set):
            reachable.add((r, c))

            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in reachable and heights[nr][nc] >= heights[r][c]:
                        dfs(nr, nc, reachable)

        # using boundaries dfs
        # starting from boundaries and going to cells that are reachable from boundaries
        for row in range(rows):
            dfs(row, 0, pacific_reachable)
            dfs(row, cols - 1, atlantic_reachable)

        for col in range(cols):
            dfs(0, col, pacific_reachable)
            dfs(rows - 1, col, atlantic_reachable)

        return list(pacific_reachable & atlantic_reachable)


if __name__ == "__main__":
    sol = Solution()

    # adj_mat = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    # print(sol.findCircleNum(adj_mat))

    # edges = [[0, 1], [1, 2], [2, 3], [4, 5]]
    # V = 7
    # print(sol.countComponents(V, edges))

    # grid = [[0, 2]]
    # print(sol.orangesRotting(grid))

    # image = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
    # print(sol.floodFill(image, 1, 1, 2))

    # adj_list = [[1, 2], [0, 4], [0, 3, 5], [2], [1, 6], [2, 6], [4, 5]]
    # print(sol.isCycle(7, adj_list))

    # mat = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
    # print(sol.updateMatrix(mat))

    # board = [
    #     ["X", "X", "X", "X"],
    #     ["X", "O", "O", "X"],
    #     ["X", "X", "O", "X"],
    #     ["X", "O", "X", "X"],
    # ]
    # sol.solve(board)
    # print(board)

    # grid = [
    #     ["1", "1", "0", "0", "0"],
    #     ["1", "1", "0", "0", "0"],
    #     ["0", "0", "1", "0", "0"],
    #     ["0", "0", "0", "1", "1"],
    # ]
    # print(sol.numIslands(grid))

    heights = [
        [1, 2, 2, 3, 5],
        [3, 2, 3, 4, 4],
        [2, 4, 5, 3, 1],
        [6, 7, 1, 4, 5],
        [5, 1, 1, 2, 4],
    ]
    print(sol.pacificAtlantic(heights))
