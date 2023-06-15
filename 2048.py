## 2048 game
from UI import UI


class Game:
    def __init__(self):
        self.GameUI = UI()

    def run2048Game(self):
        self.GameUI.runGame(600, 750)


if __name__ == "__main__":
    Game().run2048Game()
