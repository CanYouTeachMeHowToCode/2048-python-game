## 2048 game
from UI import UI

class Game(object):
    def __init__(self):
        self.size = 4 # implement customization later
        self.GameUI = UI(self.size)

    def run2048Game(self):
        self.GameUI.runGame(600, 750)

if __name__ == "__main__":
    Game().run2048Game()