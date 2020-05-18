## 2048 game AI

import random
import copy
from board import Board

class AI(object):
    def __init__(self, size, level):
        self.size = size
        self.GameBoard = Board(self.size)
        self.level = ["easy", "normal", "hard"][level]
        # self.weightBoard = 

    def move(self):
        if self.level == "easy": self.getMaxMove1()
        elif self.level == "normal": 
            step = 0
            while not self.GameBoard.GameOver():
                step += 1
                print("step:%d" % step)
                self.expectiMiniMax()
            print("Game Over")

    # using simple algorithm that only counts the current step that can reach 
    # the highest score
    def getMaxMove1(self):
        originalBoard = copy.deepcopy(self.GameBoard.board)
        originalScore = self.GameBoard.score
        assert(not self.GameBoard.GameOver())

        upScore = downScore = leftScore = rightScore = 0
        if self.GameBoard.moveUp() : upScore = self.GameBoard.score - originalScore
        self.GameBoard.board = originalBoard
        self.GameBoard.score = originalScore

        if self.GameBoard.moveDown() : downScore = self.GameBoard.score - originalScore
        self.GameBoard.board = originalBoard
        self.GameBoard.score = originalScore

        if self.GameBoard.moveLeft() : leftScore = self.GameBoard.score - originalScore
        self.GameBoard.board = originalBoard
        self.GameBoard.score = originalScore

        if self.GameBoard.moveRight() : rightScore = self.GameBoard.score - originalScore
        self.GameBoard.board = originalBoard
        self.GameBoard.score = originalScore

        scoreList = [upScore, downScore, leftScore, rightScore]
        # moving each direction has the same score
        if (len(set(scoreList)) <= 1) : action = random.randint(0, 3)
        else: action = scoreList.index(max(scoreList))

        if action == 0:
            print("moving up")
            self.GameBoard.moveUp()
        elif action == 1:
            print("moving down")
            self.GameBoard.moveDown()
        elif action == 2:
            print("moving left")
            self.GameBoard.moveLeft()
        elif action == 3:
            print("moving right")
            self.GameBoard.moveRight()
        else:
            print("should not reach here!")
            assert(False)

        self.GameBoard.addNewTile() # add a new number after each move
        self.GameBoard.printBoard()

    # using ExpectiMiniMax algorithm. 
    # Reference : http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    # def expectiMiniMax(self, board, computer, depth = 3):


# test
if __name__ == "__main__":
    easyAI = AI(4, 0)
    step = 0
    while not easyAI.GameBoard.GameOver():
        step += 1
        print("step:%d" % step)
        easyAI.move()
    print("Game Over")

