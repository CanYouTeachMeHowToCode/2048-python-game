## 2048 game
from .UI import UI

class Game(object):
    def __init__(self):
        self.GameUI = UI()

    def run2048Game(self):
        self.GameUI.runGame(600, 750)

def main():
    Game().run2048Game()

if __name__ == "__main__":
    main()