from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        Given an m x n grid of characters board and a string word, return true if word exists in the grid.
        """
        rows, cols = len(board), len(board[0])

        def dfs(r: int, c: int, i: int) -> bool:
            if len(word) == i:
                return True
            if r < 0 or c < 0 or r >= rows or c >= cols or board[r][c] != word[i]:
                return False

            temp = board[r][c]
            board[r][c] = "#"

            found = (
                dfs(r + 1, c, i + 1)
                or dfs(r - 1, c, i + 1)
                or dfs(r, c + 1, i + 1)
                or dfs(r, c - 1, i + 1)
            )

            board[r][c] = temp
            return found

        for row in range(rows):
            for col in range(cols):
                if board[row][col] == word[0] and dfs(row, col, 0):
                    return True
        return False

    def letterCombinations(self, digits: str) -> List[str]:
        """
        Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
        """
        phone = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        result = []

        def backtrack(path: str, idx: int) -> None:
            if len(digits) == idx:
                result.append(path)
                return

            for letter in phone[digits[idx]]:
                backtrack(path + letter, idx + 1)

        backtrack("", 0)
        return result

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Given an integer array nums of unique elements, return all possible subsets (the power set).
        """

        result = []

        def backtrack(path: List[int], idx: int) -> None:
            if len(nums) == idx:
                result.append(path)
                return

            backtrack(path.copy(), idx + 1)
            path.append(nums[idx])
            backtrack(path.copy(), idx + 1)

        backtrack([], 0)
        return result

    def generateParenthesis(self, n: int) -> List[str]:
        """
        Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
        """
        output = []

        def backtrack(combo: str, open: int, close: int) -> None:
            if open == n and close == n:
                output.append(combo)
                return
            if open < n:
                backtrack(combo + "(", open + 1, close)
            if close < open:
                backtrack(combo + ")", open, close + 1)

        backtrack("(", 1, 0)
        return output

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """Return combinations whose sum is equal to target."""
        res = []

        def backtrack(start_idx: int, combo: List[int], target: int) -> None:
            if target == 0:
                res.append(list(combo))
                return

            for idx in range(start_idx, len(candidates)):
                num = candidates[idx]
                if num > target:
                    return
                combo.append(num)
                backtrack(idx, combo, target - num)
                combo.pop()
            return

        candidates.sort()
        backtrack(0, [], target)
        return res

    


if __name__ == "__main__":
    sol = Solution()

    # board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    # word = "ABCB"
    # print(sol.exist(board, word))

    # print(sol.letterCombinations("23"))
    # print(sol.subsets([1, 2, 3]))
    # print(sol.generateParenthesis(3))

    candidates = [2, 3, 6, 7]
    target = 7
    print(sol.combinationSum(candidates, target))
