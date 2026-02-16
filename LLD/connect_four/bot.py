from .game import Game


class BotEngine:
    def choose_move(self, game: Game) -> int:
        board = game.get_board()
        for c in range(board.get_cols()):
            if board.can_place(c):
                return c
        return -1
