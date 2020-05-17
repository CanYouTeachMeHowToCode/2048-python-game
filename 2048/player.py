## 2048 game player

import random

class Player(object):
    def __init__(self):


    def printBoard(self):
        board = self.board
        print("current board:\n")
        for i in range(self.col):
            print(board[i], "\n")

    def move(self, direction):
        




# test
if __name__ == "__main__":
    board = Board(4, 4)
    board.printBoard()