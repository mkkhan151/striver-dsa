from typing import List


class Solution:
    def climbStairs(self, n: int) -> int:
        """
        Given n number of stairs.
        Returns the distinct ways to climb to the top of stairs.
        Each time 1 or 2 steps can be climbed.
        """

        # using top-down Approach - Memoization
        # devide problem into sub-problems and cache the solution of sub-problems on first calculation
        # memo = {}  # stores the ans for sub-solutions (overlaping sub-problems)

        # def climb_helper(i: int) -> int:
        #     if i <= 1:
        #         return 1

        #     # check if value is already in cache before making recursive calls
        #     # if once calculated for i steps no need to calculate again
        #     if i in memo:
        #         return memo[i]

        #     # not calculated for i step, store result in cache before returning it
        #     memo[i] = climb_helper(i - 1) + climb_helper(i - 2)
        #     return memo[i]

        # return climb_helper(n)

        # using bottom-up approach
        # We know the base cases that for 0 and 1 stairs there is only 1 way
        # If we know for n - 1 and n - 2 then we can calculate for n. Here n = (n - 1) + (n - 2)
        if n <= 1:
            return 1
        # dp = [0] * (n + 1)

        # # Base cases
        # dp[0] = 1
        # dp[1] = 1

        # for i in range(2, n + 1):
        #     dp[i] = dp[i - 1] + dp[i - 2]
        # return dp[n]

        # Or just use three variables
        prev, prev_next = 1, 1
        curr = prev + prev_next
        for i in range(n - 1):
            curr = prev + prev_next
            prev = prev_next
            prev_next = curr
        return curr

    def rob(self, nums: List[int]) -> int:
        """
        Given an integer array nums representing the amount of money of each house.
        Return the maximum amount of money you can rob tonight without alerting the police.
        Police is alerted when you take from adjacent houses e.g house 2 and 3
        """
        # using top-down (memoization) approach
        # memo = {}

        # def rob_helper(i: int) -> int:
        #     if i == 0:
        #         return 0
        #     if i == 1:
        #         return nums[0]

        #     if i in memo:
        #         return memo[i]

        #     take = rob_helper(i - 2) + nums[i - 1]
        #     skip = rob_helper(i - 1)
        #     memo[i] = max(skip, take)
        #     return memo[i]

        # return rob_helper(len(nums))

        # Using bottom-up approach
        # if not nums:
        #     return 0

        # # Initilize the dp array
        # dp = [0] * (len(nums) + 1)

        # # fill in base cases (dp[0] = 0 already)
        # dp[1] = nums[0]

        # # iterate to fill in the rest of dp array
        # for i in range(2, len(nums) + 1):
        #     # fill in dp[i] using recurrence relation
        #     take = dp[i - 2] + nums[i - 1]
        #     skip = dp[i - 1]
        #     dp[i] = max(skip, take)

        # return dp[len(nums)]

        # Further optimization
        # Looking at the recurrence relation
        # dp(i) = max(dp(i - 1), dp(i - 2) + treasure[i - 1])
        # We only need two variables to solve this
        if not nums:
            return 0

        prev, curr = 0, nums[0]

        for i in range(2, len(nums) + 1):
            # calculate the next value of dp
            take = prev + nums[i - 1]
            skip = curr
            prev, curr = curr, max(skip, take)
        return curr


if __name__ == "__main__":
    sol = Solution()

    # print(sol.climbStairs(4))

    nums = [1, 2, 3, 1]
    print(sol.rob(nums))
