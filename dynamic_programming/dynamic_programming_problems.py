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


if __name__ == "__main__":
    sol = Solution()

    print(sol.countBits(0))
