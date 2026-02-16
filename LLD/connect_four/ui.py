import tkinter as tk

import customtkinter

from .board import DiscColor
from .bot import BotEngine
from .game import Game, GameState
from .player import Player

# Colors
BOARD_BG = "#0d3b66"
CELL_EMPTY = "#1a1a2e"
YELLOW_DISC = "#f9c74f"
RED_DISC = "#e63946"
HOVER_COLOR = "#2a9d8f"
FRAME_BG = "#16213e"
APP_BG = "#1a1a2e"
STATUS_TEXT = "#a8dadc"
BTN_PRIMARY = "#e63946"
BTN_SECONDARY = "#457b9d"


class UIManager:
    CELL_SIZE = 80
    DISC_PADDING = 8
    ROWS = 6
    COLS = 7
    BOARD_W = COLS * CELL_SIZE
    BOARD_H = ROWS * CELL_SIZE

    def __init__(self) -> None:
        self._root = customtkinter.CTk()
        self._root.title("Connect Four")
        self._root.resizable(False, False)
        self._root.configure(fg_color=APP_BG)

        # Game state
        self._game: Game | None = None
        self._player1: Player | None = None
        self._player2: Player | None = None
        self._bot: BotEngine | None = None
        self._mode: str = "hvb"
        self._game_generation: int = 0

        # Canvas tracking
        self._cell_canvas_ids: dict[tuple[int, int], int] = {}
        self._col_hover: int | None = None
        self._col_buttons: list[customtkinter.CTkButton] = []

        # Build screens
        self._start_frame = customtkinter.CTkFrame(self._root, fg_color=APP_BG)
        self._game_frame = customtkinter.CTkFrame(self._root, fg_color=FRAME_BG)
        self._result_overlay: customtkinter.CTkFrame | None = None

        self._build_start_screen()
        self._build_game_screen()
        self._show_start_screen()

    # ── Screen builders ──────────────────────────────────────────

    def _build_start_screen(self) -> None:
        container = customtkinter.CTkFrame(self._start_frame, fg_color="transparent")
        container.place(relx=0.5, rely=0.5, anchor="center")

        customtkinter.CTkLabel(
            container,
            text="CONNECT FOUR",
            font=("Helvetica", 36, "bold"),
            text_color=YELLOW_DISC,
        ).pack(pady=(0, 40))

        customtkinter.CTkButton(
            container,
            text="Human vs Bot",
            width=220,
            height=45,
            corner_radius=12,
            fg_color=BTN_PRIMARY,
            hover_color="#c1121f",
            font=("Helvetica", 16),
            command=lambda: self._start_game("hvb"),
        ).pack(pady=8)

        customtkinter.CTkButton(
            container,
            text="Human vs Human",
            width=220,
            height=45,
            corner_radius=12,
            fg_color=BTN_SECONDARY,
            hover_color=HOVER_COLOR,
            font=("Helvetica", 16),
            command=lambda: self._start_game("hvh"),
        ).pack(pady=8)

    def _build_game_screen(self) -> None:
        # Top bar
        top_bar = customtkinter.CTkFrame(self._game_frame, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        top_bar.columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            top_bar,
            text="CONNECT FOUR",
            font=("Helvetica", 20, "bold"),
            text_color=YELLOW_DISC,
        ).grid(row=0, column=0, sticky="w")

        btn_frame = customtkinter.CTkFrame(top_bar, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        customtkinter.CTkButton(
            btn_frame,
            text="New Game",
            width=100,
            height=32,
            corner_radius=8,
            fg_color=BTN_PRIMARY,
            hover_color="#c1121f",
            font=("Helvetica", 13),
            command=self._reset_game,
        ).pack(side="left", padx=4)

        customtkinter.CTkButton(
            btn_frame,
            text="Menu",
            width=80,
            height=32,
            corner_radius=8,
            fg_color=BTN_SECONDARY,
            hover_color=HOVER_COLOR,
            font=("Helvetica", 13),
            command=self._show_start_screen,
        ).pack(side="left", padx=4)

        # Status label
        self._status_label = customtkinter.CTkLabel(
            self._game_frame,
            text="",
            font=("Helvetica", 16),
            text_color=STATUS_TEXT,
        )
        self._status_label.grid(row=1, column=0, pady=6)

        # Column buttons
        col_buttons_frame = customtkinter.CTkFrame(
            self._game_frame, fg_color="transparent"
        )
        col_buttons_frame.grid(row=2, column=0, pady=(0, 4))
        self._col_buttons = []
        for c in range(self.COLS):
            btn = customtkinter.CTkButton(
                col_buttons_frame,
                text="\u25bc",
                width=self.CELL_SIZE - 4,
                height=28,
                corner_radius=6,
                fg_color=BTN_SECONDARY,
                hover_color=HOVER_COLOR,
                font=("Helvetica", 14),
                command=lambda col=c: self._handle_human_move(col),
            )
            btn.grid(row=0, column=c, padx=2)
            self._col_buttons.append(btn)

        # Board canvas
        self._board_canvas = tk.Canvas(
            self._game_frame,
            width=self.BOARD_W,
            height=self.BOARD_H,
            bg=BOARD_BG,
            highlightthickness=0,
        )
        self._board_canvas.grid(row=3, column=0, padx=10, pady=(0, 10))
        self._board_canvas.bind("<Button-1>", self._on_canvas_click)
        self._board_canvas.bind("<Motion>", self._on_canvas_motion)
        self._board_canvas.bind("<Leave>", self._on_canvas_leave)

        # Draw empty cells
        self._draw_board()

    # ── Screen transitions ───────────────────────────────────────

    def _show_start_screen(self) -> None:
        self._game_frame.pack_forget()
        self._hide_result_overlay()
        self._start_frame.pack(fill="both", expand=True)
        self._root.geometry("400x300")

    def _show_game_screen(self) -> None:
        self._start_frame.pack_forget()
        self._game_frame.pack(fill="both", expand=True)
        w = self.BOARD_W + 20
        h = self.BOARD_H + 140
        self._root.geometry(f"{w}x{h}")

    # ── Game init / reset ────────────────────────────────────────

    def _start_game(self, mode: str) -> None:
        self._mode = mode
        self._player1 = Player("Player 1", DiscColor.YELLOW)
        if mode == "hvb":
            self._player2 = Player("Bot", DiscColor.RED)
            self._bot = BotEngine()
        else:
            self._player2 = Player("Player 2", DiscColor.RED)
            self._bot = None

        self._game_generation += 1
        self._game = Game(self._player1, self._player2)
        self._refresh_board()
        self._update_status()
        self._hide_result_overlay()
        self._show_game_screen()

    def _reset_game(self) -> None:
        self._game_generation += 1
        self._game = Game(self._player1, self._player2)
        self._refresh_board()
        self._update_status()
        self._hide_result_overlay()
        self._set_board_interactive(True)

    # ── Board rendering ──────────────────────────────────────────

    def _draw_board(self) -> None:
        self._cell_canvas_ids.clear()
        pad = self.DISC_PADDING
        for row in range(self.ROWS):
            for col in range(self.COLS):
                x0 = col * self.CELL_SIZE + pad
                y0 = row * self.CELL_SIZE + pad
                x1 = (col + 1) * self.CELL_SIZE - pad
                y1 = (row + 1) * self.CELL_SIZE - pad
                oval_id = self._board_canvas.create_oval(
                    x0, y0, x1, y1, fill=CELL_EMPTY, outline=""
                )
                self._cell_canvas_ids[(row, col)] = oval_id

    def _refresh_board(self) -> None:
        if self._game is None:
            return
        board = self._game.get_board()
        for row in range(board.get_rows()):
            for col in range(board.get_cols()):
                self._paint_cell(row, col)

    def _paint_cell(self, row: int, col: int) -> None:
        disc = self._game.get_board().get_cell(row, col)
        if disc == DiscColor.YELLOW:
            fill = YELLOW_DISC
        elif disc == DiscColor.RED:
            fill = RED_DISC
        else:
            fill = CELL_EMPTY
        oval_id = self._cell_canvas_ids[(row, col)]
        self._board_canvas.itemconfig(oval_id, fill=fill)

    # ── Hover highlight ──────────────────────────────────────────

    def _on_canvas_motion(self, event: tk.Event) -> None:
        col = min(max(event.x // self.CELL_SIZE, 0), self.COLS - 1)
        if col == self._col_hover:
            return
        self._clear_column_highlight()
        self._col_hover = col
        for row in range(self.ROWS):
            oval_id = self._cell_canvas_ids[(row, col)]
            self._board_canvas.itemconfig(oval_id, outline=HOVER_COLOR, width=3)

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self._clear_column_highlight()

    def _clear_column_highlight(self) -> None:
        if self._col_hover is not None:
            for row in range(self.ROWS):
                oval_id = self._cell_canvas_ids[(row, self._col_hover)]
                self._board_canvas.itemconfig(oval_id, outline="", width=0)
            self._col_hover = None

    # ── Event handlers ───────────────────────────────────────────

    def _on_canvas_click(self, event: tk.Event) -> None:
        col = min(max(event.x // self.CELL_SIZE, 0), self.COLS - 1)
        self._handle_human_move(col)

    def _handle_human_move(self, col: int) -> None:
        if self._game is None:
            return
        if self._game.get_game_state() != GameState.IN_PROGRESS:
            return

        current = self._game.get_current_player()
        # In HvB mode, ignore clicks during bot's turn
        if self._mode == "hvb" and current is self._player2:
            return

        if not self._game.make_move(current, col):
            return

        self._after_move()

    def _after_move(self) -> None:
        self._refresh_board()
        self._update_status()

        state = self._game.get_game_state()
        if state in (GameState.WON, GameState.DRAW):
            self._set_board_interactive(False)
            self._show_result_overlay()
            return

        # Schedule bot move if needed
        if self._mode == "hvb" and self._game.get_current_player() is self._player2:
            self._schedule_bot_move()

    def _schedule_bot_move(self) -> None:
        gen = self._game_generation
        self._set_board_interactive(False)
        self._root.after(500, lambda: self._handle_bot_move(gen))

    def _handle_bot_move(self, gen: int) -> None:
        # Discard stale callback
        if gen != self._game_generation:
            return
        if self._game is None or self._game.get_game_state() != GameState.IN_PROGRESS:
            return

        col = self._bot.choose_move(self._game)
        if col == -1:
            return
        self._game.make_move(self._player2, col)
        self._set_board_interactive(True)
        self._after_move()

    # ── Status / result display ──────────────────────────────────

    def _update_status(self) -> None:
        if self._game is None:
            return
        state = self._game.get_game_state()
        if state == GameState.IN_PROGRESS:
            player = self._game.get_current_player()
            color_word = (
                "Yellow" if player.get_color() == DiscColor.YELLOW else "Red"
            )
            self._status_label.configure(
                text=f"{player.get_name()}'s Turn  ({color_word})",
                text_color=YELLOW_DISC
                if player.get_color() == DiscColor.YELLOW
                else RED_DISC,
            )
        elif state == GameState.WON:
            winner = self._game.get_winner()
            self._status_label.configure(
                text=f"{winner.get_name()} Wins!",
                text_color=YELLOW_DISC,
            )
        else:
            self._status_label.configure(
                text="It's a Draw!",
                text_color=STATUS_TEXT,
            )

    def _show_result_overlay(self) -> None:
        self._hide_result_overlay()

        self._result_overlay = customtkinter.CTkFrame(
            self._game_frame, fg_color=APP_BG, corner_radius=16
        )
        self._result_overlay.place(
            relx=0.5, rely=0.55, anchor="center", relwidth=0.7, relheight=0.35
        )

        state = self._game.get_game_state()
        if state == GameState.WON:
            winner = self._game.get_winner()
            color = (
                YELLOW_DISC
                if winner.get_color() == DiscColor.YELLOW
                else RED_DISC
            )
            text = f"{winner.get_name()} Wins!"
        else:
            color = STATUS_TEXT
            text = "It's a Draw!"

        customtkinter.CTkLabel(
            self._result_overlay,
            text=text,
            font=("Helvetica", 28, "bold"),
            text_color=color,
        ).pack(pady=(20, 16))

        btn_row = customtkinter.CTkFrame(self._result_overlay, fg_color="transparent")
        btn_row.pack()

        customtkinter.CTkButton(
            btn_row,
            text="Play Again",
            width=130,
            height=36,
            corner_radius=10,
            fg_color=BTN_PRIMARY,
            hover_color="#c1121f",
            font=("Helvetica", 14),
            command=self._reset_game,
        ).pack(side="left", padx=6)

        customtkinter.CTkButton(
            btn_row,
            text="Main Menu",
            width=130,
            height=36,
            corner_radius=10,
            fg_color=BTN_SECONDARY,
            hover_color=HOVER_COLOR,
            font=("Helvetica", 14),
            command=self._show_start_screen,
        ).pack(side="left", padx=6)

    def _hide_result_overlay(self) -> None:
        if self._result_overlay is not None:
            self._result_overlay.destroy()
            self._result_overlay = None

    def _set_board_interactive(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for btn in self._col_buttons:
            btn.configure(state=state)

    # ── Run ──────────────────────────────────────────────────────

    def run(self) -> None:
        self._root.mainloop()
