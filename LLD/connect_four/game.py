from enum import StrEnum

from .board import Board
from .player import Player


class GameState(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    WON = "WON"
    DRAW = "DRAW"


class Game:
    def __init__(self, player_1: Player, player_2: Player) -> None:
        self.board = Board()
        self.player_1 = player_1
        self.player_2 = player_2
        self.current_player = player_1
        self.state: GameState = GameState.IN_PROGRESS
        self.winner: Player | None = None

    def make_move(self, player: Player, column: int) -> bool:
        if self.state != GameState.IN_PROGRESS:
            return False

        if player is not self.current_player:
            return False

        row = self.board.place_disc(column, player.get_color())
        if row == -1:
            return False

        if self.board.check_win(row, column, player.get_color()):
            self.state = GameState.WON
            self.winner = player
        elif self.board.is_full():
            self.state = GameState.DRAW
        else:
            self.current_player = (
                self.player_2 if player is self.player_1 else self.player_1
            )
        return True

    def get_current_player(self) -> Player:
        return self.current_player

    def get_game_state(self) -> GameState:
        return self.state

    def get_winner(self) -> Player | None:
        return self.winner

    def get_board(self) -> Board:
        return self.board
