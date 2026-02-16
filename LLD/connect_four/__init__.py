# from .board import DiscColor
# from .bot import BotEngine
# from .game import Game, GameState
# from .player import Player

# human_player = Player("Kamran", DiscColor.YELLOW)
# bot_player = Player("Bot", DiscColor.RED)

# game = Game(human_player, bot_player)
# engine = BotEngine()

# while game.get_game_state() == GameState.IN_PROGRESS:
#     curr_player = game.get_current_player()

#     if curr_player is human_player:
#         # we may take input from UI
#         col = input("Choose column: ")
#         col = int(col)
#     else:
#         col = engine.choose_move(game)
#     game.make_move(curr_player, col)
