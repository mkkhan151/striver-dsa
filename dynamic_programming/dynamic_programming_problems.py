from typing import List


class Solution:
    def countBits(self, n: int) -> List[int]:
        """
        Returns the number of 1's in binary form of each number from 0 to n (n is given)
        """
        dp = [0] * (n + 1)  # dp[0] = 0 is the base case

        for i in range(2, n + 1):
            dp[i] = dp[i // 2] + (i % 2)
        return dp

    def numDecodings(self, s: str) -> int:
        """
        Given a string s containing only digits, return the number of ways to decode it.
        "1" -> 'A'
        "2" -> 'B'
        ...
        "25" -> 'Y'
        "26" -> 'Z'
        """
        if not s or s[0] == "0":
            return 0

        n = len(s)
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1

        for i in range(2, n + 1):
            digit = int(s[i - 1])
            if digit != 0:
                dp[i] += dp[i - 1]

            digit = int(s[i - 2 : i])
            if 10 <= digit <= 26:
                dp[i] += dp[i - 2]
        return dp[n]

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        """
        Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
        """
        if not matrix:
            return 0
        r = len(matrix)
        c = len(matrix[0])
        dp = [[0] * (c + 1) for _ in range(r + 1)]
        max_side = 0
        for i in range(1, r + 1):
            for j in range(1, c + 1):
                if matrix[i - 1][j - 1] == "1":
                    top = dp[i - 1][j]
                    left = dp[i][j - 1]
                    diag = dp[i - 1][j - 1]
                    dp[i][j] = min(top, left, diag) + 1
                    max_side = max(max_side, dp[i][j])
        return max_side * max_side

    def uniquePaths(self, m: int, n: int) -> int:
        """
        Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner from Top-left corner.
        """
        # Using Tab-Down approach.
        # if m == 1 or n == 1:
        #     return 1
        # return self.uniquePaths(m - 1, n) + self.uniquePaths(m, n - 1)

        # with memoization
        # memo = {}

        # def uniquePathsHelper(m: int, n: int) -> int:
        #     if m == 1 or n == 1:
        #         return 1

        #     if (m, n) in memo:
        #         return memo[(m, n)]

        #     memo[(m, n)] = uniquePathsHelper(m - 1, n) + uniquePathsHelper(m, n - 1)
        #     return memo[(m, n)]

        # return uniquePathsHelper(m, n)

        # With bottom-up approach
        dp = [[0] * n for _ in range(m)]

        # base cases first row and column are always 1
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1

        # Fill the rest of the dp array
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]


if __name__ == "__main__":
    sol = Solution()

    # print(sol.countBits(0))
    # print(sol.numDecodings("1234"))

    # mat = [
    #     ["1", "0", "1", "0", "0"],
    #     ["1", "0", "1", "1", "1"],
    #     ["1", "1", "1", "1", "1"],
    #     ["1", "0", "0", "1", "0"],
    # ]
    # print(sol.maximalSquare(mat))
    print(sol.uniquePaths(2, 4))
