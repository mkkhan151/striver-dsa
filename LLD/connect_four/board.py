from enum import StrEnum
from typing import List


class DiscColor(StrEnum):
    YELLOW = "yellow"
    RED = "red"


class Board:
    def __init__(self) -> None:
        self.rows = 6
        self.cols = 7
        self.grid: List[List[DiscColor | None]] = [
            [None] * self.cols for _ in range(self.rows)
        ]

    def get_rows(self) -> int:
        return self.rows

    def get_cols(self) -> int:
        return self.cols

    def can_place(self, col: int) -> bool:
        return 0 <= col < self.cols and self.grid[0][col] is None

    def place_disc(self, col: int, color: DiscColor) -> int:
        if not self.can_place(col):
            return -1

        row = self.rows - 1
        while self.grid[row][col] is not None:
            row -= 1
        self.grid[row][col] = color
        return row

    def is_full(self) -> bool:
        # Full if all entries in first row are marked
        # return all([i is not None for i in self.grid[0]])
        for c in range(self.cols):
            if self.can_place(c):
                return False
        return True

    def __in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def __count_in_direction(
        self, row: int, col: int, dr: int, dc: int, color: DiscColor
    ) -> int:
        count = 0
        r = row + dr
        c = col + dc
        while self.__in_bounds(r, c) and self.grid[r][c] == color:
            count += 1
            r += dr
            c += dc
        return count

    def check_win(self, row: int, col: int, color: DiscColor) -> bool:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False

        if color != self.grid[row][col]:
            return False

        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        for dr, dc in directions:
            count = 1
            count += self.__count_in_direction(
                row, col, dr, dc, color
            )  # move in direction
            count += self.__count_in_direction(
                row, col, -dr, -dc, color
            )  # move in opposite direction
            if count >= 4:
                return True
        return False

    def get_cell(self, row: int, col: int) -> DiscColor | None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None
