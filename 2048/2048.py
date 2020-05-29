## 2048 game
from UI import UI
# from board import Board

class Game(object):
    def __init__(self):
        self.GameUI = UI()
        # self.size = 4 # implement customization later
        # self.GameBoard = Board(self.size)
        # self.GameUI = UI(self.GameBoard)

    def run2048Game(self):
        self.GameUI.runGame(600, 750)

if __name__ == "__main__":
    Game().run2048Game()