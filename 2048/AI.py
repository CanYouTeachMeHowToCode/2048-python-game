## 2048 game AI

import random
import copy
from board import Board

class AI(object):
    def __init__(self, size, level):
        self.size = size
        self.GameBoard = Board(self.size)
        self.level = ["easy", "normal", "hard"][level]

        # weight board assign the grids on board with weight 
        # in zigzag order increasing exponentially with base 4
        # e.g. for a weight board with size 4, the board weight is 
            # [[4^0,   4^1,  4^2,   4^3],
            #  [4^7,   4^6,  4^5,   4^4],
            #  [4^8,   4^9,  4^10, 4^11],
            #  [4^15, 4^14,  4^13, 4^12]]
        # Reference: http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
        def weightBoard(size):
            board = [[(row * size + col) for col in range(size)] for row in range(size)]
            for row in range(size):
                if row % 2 : board[row] = board[row][::-1]
            for row in range(size):
                for col in range(size):
                    exp = board[row][col]
                    board[row][col] = 4 ** exp
            return board
        self.weightBoard = weightBoard(self.size)
        # print("weightBoard :", self.weightBoard)

    def nextMove(self):
        if self.level == "easy": self.getMaxMove1()
        elif self.level == "normal": self.getMaxMove2()

    def performAction(self, board, action):
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

    # using simple algorithm that only counts the current step that can reach 
    # the highest score
    def getMaxMove1(self):
        assert(not self.GameBoard.GameOver())
        # maxie move
        originalBoard = copy.deepcopy(self.GameBoard.board)
        originalScore = self.GameBoard.score

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
        self.performAction(action)

        # (fake computer minie move that just uses the normal method to add new numbers)
        self.GameBoard.addNewTile() # add a new number after each move

    # using ExpectiMiniMax algorithm. 
    # Reference : http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    # the evaluate function estimates the current situation on the board and 
    # return a score that quantifies the situation. The evaluation algorithm
    # is that the score is equal to the sum of the product of weight of a 
    # certain tile and the number on it. 
    # (i.e. ∑(row)∑(col) weightBoard[row][col] * GameBoard[row][col])
    def evaluate(self):
        score = 0
        for row in range(self.size):
            for col in range(self.size):
                score += (self.weightBoard[row][col] * self.GameBoard.board[row][col])
        return score

    # the method for computer that does not add the numbers "normally" --
    # add the number based on the current game board situation
    def addNewNum(self, board, action):
        index = action[0]
        addNum = action[1]
        assert(board[index[0]][index[1]] == 0)
        board[index[0]][index[1]] = addNum

    def expectiMiniMax(self, board, playerMove, opponent, depth):
        if playerMove : # player's move (moving up, down, left or right)
            if not depth: return (self.evaluate(), None) # depth = 0

            # get all legal actions and preserve the board
            originalBoard = copy.deepcopy(self.GameBoard.board)
            originalScore = self.GameBoard.score

            actions = []
            if self.GameBoard.moveUp() : actions.append(0)
            self.GameBoard.board = originalBoard
            self.GameBoard.score = originalScore

            if self.GameBoard.moveDown() : actions.append(1)
            self.GameBoard.board = originalBoard
            self.GameBoard.score = originalScore

            if self.GameBoard.moveLeft() : actions.append(2)
            self.GameBoard.board = originalBoard
            self.GameBoard.score = originalScore

            if self.GameBoard.moveRight() : actions.append(3)
            self.GameBoard.board = originalBoard
            self.GameBoard.score = originalScore

            print("actions: ", end = "")
            print(actions)
            if not actions: return (self.evaluate(), None) # no legal actions

            (bestScore, bestAction) = (-float('inf'), None)

            for action in actions:
                beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
                self.performAction(action)
                (computerScore, computerAction) = opponent.expectiMiniMax(not playerMove, self, depth-1)
                self.GameBoard.board = beforeMoveBoard

                playerScore = -computerScore
                if playerScore > bestScore:
                    bestScore = playerScore
                    bestAction = action

        else : # computer's move (put a new number on the tile to make the player
               # solve the board and get higher scores as hard as possible)
            if not depth: return (self.evaluate(), None) # depth = 0

            # even though the real computer will put the new numbers randomly,
            # we still assume that it can put 2 or 4 on any empty tile as it 
            # wishes to make the board harder for player to solve.
            emptyTiles = [] # tuple list => empty tile coordinates
            for i in range(self.size):
                for j in range(self.size):
                    if not self.GameBoard.board[i][j]: # this tile is empty
                        emptyTiles.append((i, j))

            actions = []
            for index in emptyTiles:
                # can add 2 or 4 on any empty tile
                actions.append((index, 2))
                actions.append((index, 4))
            print("actions: ", end = "")
            print(actions)
            if not actions: return (self.evaluate(), None) # no legal actions

            (bestScore, bestAction) = (-float('inf'), None)

            for action in actions:
                beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
                self.addNewNum(action) # perform computer's action
                (playerScore, playerAction) = self.expectiMiniMax(not playerMove, opponent, depth-1)
                self.GameBoard.board = beforeMoveBoard

                computerScore = -playerScore
                if computerScore > bestScore:
                    bestScore = computerScore
                    bestAction = action

        return (bestScore, bestAction)

    def getMaxMove2(self):
        computer = AI(self.size, 1)
        (score, action) = self.expectiMiniMax(True, computer, 1)
        print("bestScore, bestAction:", (score, action))
        # action = self.expectiMiniMax(computer, 3)[1]
        self.performAction(action)

    def playTheGame(self):
        step = 0
        while not self.GameBoard.GameOver():
            step += 1
            print("step:%d" % step)
            self.nextMove()
            self.GameBoard.printBoard()
        print("Game Over")


# test
if __name__ == "__main__":
    # easyAI = AI(4, 0)
    # easyAI.playTheGame()

    normalAI = AI(4, 1)
    normalAI.playTheGame()


