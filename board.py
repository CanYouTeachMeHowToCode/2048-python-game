## 2048 game board

import random
import copy
import numpy as np


class Board:
    def __init__(self, size=4):
        # size default set to 4, but can be customized
        self.empty = 0  # 0 represents empty grid
        self.board = np.array([[self.empty for _ in range(size)] for _ in range(size)])
        self.size = size
        self.directionList = ["Up", "Down", "Left", "Right"]
        self.score = 0  # initial score is 0

        # start coordinates (random)
        startIndex = (random.randint(0, size - 1), random.randint(0, size - 1))
        # start number can be 2 or 4, 2 occurs 80% and 4 occurs 20%
        startNum = random.choice([2, 2, 2, 2, 2, 2, 2, 2, 4, 4])
        self.board[startIndex[0]][startIndex[1]] = startNum

    def printBoard(self):
        print("\ncurrent board:")
        print(self.board)
        print("\n")

    def findAllEmptyTiles(self):
        return list(zip(*np.where(self.board == self.empty)))

    def addNewTile(self):
        emptyTiles = self.findAllEmptyTiles()
        addIndex = random.choice(emptyTiles)
        # 2 occurs 80% and 4 occurs 20%
        addNum = random.choice([2, 2, 2, 2, 2, 2, 2, 2, 4, 4])
        self.board[addIndex[0]][addIndex[1]] = addNum
        return addIndex, addNum

    def isSameBoard(self, board1, board2):
        return np.array_equal(board1, board2)

    def moveLeft(self):
        origBoard = copy.deepcopy(self.board)
        for row in range(self.size):
            if list(
                filter(lambda x: x != self.empty, self.board[row])
            ):  # ignore empty rows
                # merge two same numbers, prior left
                col = 0
                while col < self.size - 1:
                    for nextCol in range(col + 1, self.size):
                        if (
                            (self.board[row][col] != self.empty)  # a number
                            and (
                                self.board[row][col] == self.board[row][nextCol]
                            )  # the tiles between the two same numbers are all empty
                            and (
                                not list(
                                    filter(
                                        lambda x: x != self.empty,
                                        self.board[row][(col + 1) : nextCol],
                                    )
                                )
                            )
                        ):  # or there're no tiles between them (i.e. two numbers are consecutive)
                            currNum = self.board[row][col]
                            self.board[row][col] = currNum * 2
                            self.score += self.board[row][col]
                            self.board[row][nextCol] = self.empty
                            col = nextCol
                            break
                    col += 1

                # move numbers to left all empty tiles on the left
                nums = list(filter(lambda x: x != self.empty, self.board[row]))
                newRow = nums + [0] * (self.size - len(nums))
                self.board[row] = newRow

        # return False if this move cannot be processed (i.e. the board after moving is the same as the original one), else return True
        return not self.isSameBoard(origBoard, self.board)

    """
    If we rotate the board 90˚ clockwise, move left and rotate 90° counterclockwise back, 
    the new board is equivalent to the original board moving down. With this trick, we can
    only use rotate and move left to implement move down, move up and move right.
    """

    # rotate the board 90° clockwise is equivalent to transpose the board and mirror the board along the vertical axis
    def transpose(self):
        self.board = np.transpose(self.board)

    def mirror(self):
        self.board = np.flip(self.board, 1)

    def rotate90Clockwise(self):
        self.transpose()
        self.mirror()

    def rotate180(self):
        for _ in range(2):
            self.transpose()
            self.mirror()

    def rotate90CounterClockwise(self):
        for _ in range(3):
            self.transpose()
            self.mirror()

    def moveDown(self):
        self.rotate90Clockwise()
        canMove = self.moveLeft()
        self.rotate90CounterClockwise()
        return canMove

    def moveUp(self):
        self.rotate90CounterClockwise()
        canMove = self.moveLeft()
        self.rotate90Clockwise()
        return canMove

    def moveRight(self):
        self.rotate180()
        canMove = self.moveLeft()
        self.rotate180()
        return canMove

    def reaches2048(self):  # have numbers greater than or equal to 2048
        return np.any(self.board >= 2048)

    def getLargestTileNumber(self):
        return np.amax(self.board)

    """
    The game is over once the player cannot make any legal move on the board or
    reach the number 2048 (edit: reach 2048 does not necessarily win the game, the
    player decides whether to continue the game or not)
    """

    def gameOver(self, verbose=True):
        originalScore = self.score

        # the player get 2048
        if self.reaches2048():
            if verbose:
                print("\nCongratulations! you get 2048 and win!\n")
            # return True

        # the player cannot make any legal move before getting 2048
        boardU = copy.deepcopy(self.board)
        boardD = copy.deepcopy(self.board)
        boardL = copy.deepcopy(self.board)
        boardR = copy.deepcopy(self.board)

        canMoveUp = self.moveUp()
        self.board = boardU
        self.score = originalScore

        canMoveDown = self.moveDown()
        self.board = boardD
        self.score = originalScore

        canMoveLeft = self.moveLeft()
        self.board = boardL
        self.score = originalScore

        canMoveRight = self.moveRight()
        self.board = boardR
        self.score = originalScore

        if not (canMoveUp or canMoveDown or canMoveLeft or canMoveRight):
            if verbose:
                print("\nYou don't have any legal moves! Game Over!")
            return True
        else:
            return False

    # this function is used for terminal playing version
    def move(self):
        while 1:
            direction = input("moving direction (Up, Down, Left, Right): ")
            canMove = False
            if direction not in self.directionList:
                print("illegal move")
            else:
                print("direction:", direction)
                if direction == "Up":
                    canMoveUp = self.moveUp()
                    canMove = canMoveUp
                elif direction == "Down":
                    canMoveDown = self.moveDown()
                    canMove = canMoveDown
                elif direction == "Left":
                    canMoveLeft = self.moveLeft()
                    canMove = canMoveLeft
                elif direction == "Right":
                    canMoveRight = self.moveRight()
                    canMove = canMoveRight
                # add a new number after each legal move
                if canMove:
                    self.addNewTile()
                else:
                    print("cannot move in this direction")
            self.printBoard()
            if self.gameOver():
                print("Game Over!\n")
                break
        print("reach here after game is over!")
