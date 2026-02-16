from .board import DiscColor


class Player:
    def __init__(self, name: str, color: DiscColor) -> None:
        self.name = name
        self.color = color

    def get_name(self) -> str:
        return self.name

    def get_color(self) -> DiscColor:
        return self.color
