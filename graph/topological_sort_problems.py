from collections import defaultdict, deque
from typing import List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        Return true if user can finish all courses else false
        """
        # Using BFS/Kahn's algorithm
        indegrees = [0] * numCourses
        adj_list = defaultdict(list)

        for src, dest in prerequisites:
            adj_list[src].append(dest)
            indegrees[dest] += 1

        q = deque([i for i in range(numCourses) if indegrees[i] == 0])
        count = 0
        while q:
            course = q.popleft()
            count += 1

            for neighbor in adj_list[course]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    q.append(neighbor)
        return count == numCourses

        # TODO: Using DFS

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """Return the list of courses in order they are possible to study."""
        in_degrees = [0] * numCourses
        adj_list = defaultdict(list)

        for dest, src in prerequisites:
            adj_list[src].append(dest)
            in_degrees[dest] += 1

        q = deque([i for i in range(numCourses) if in_degrees[i] == 0])
        order = []
        while q:
            course = q.popleft()
            order.append(course)
            for neighbor in adj_list[course]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    q.append(neighbor)

        return order if numCourses == len(order) else []


if __name__ == "__main__":
    sol = Solution()

    numCourses = 1
    prerequisites = []
    # print(sol.canFinish(numCourses, prerequisites))
    print(sol.findOrder(numCourses, prerequisites))
