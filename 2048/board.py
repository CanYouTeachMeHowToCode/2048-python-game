## 2048 game board

import random
import copy

class Board(object):
    def __init__(self, size=4):
        # size default set to 4, but can be customized
        self.empty = 0 # 0 represents empty grid
        self.board = [[self.empty for _ in range(size)] for _ in range(size)]
        self.size = size
        self.directionList = ["Up", "Down", "Left", "Right"]
        self.score = 0 # initial score is 0

        # start coordinates 
        startIndex = (random.randint(0, size-1), random.randint(0, size-1)) 
        # start number can be 2 or 4
        # 2 occurs 80% and 4 occurs 20%
        startNum = random.choice([2, 2, 2, 2, 2, 2, 2, 4, 4, 2])

        self.board[startIndex[0]][startIndex[1]] = startNum

    def printBoard(self):
        board = self.board
        print("current board:\n")
        for i in range(self.size):
            print(board[i], "\n")

    def addNewTile(self):
        emptyTiles = [] # tuple list => empty tile coordinates
        for i in range(self.size):
            for j in range(self.size):
                if not self.board[i][j]: # this tile is empty
                    emptyTiles.append((i, j))
        addIndex = random.choice(emptyTiles)
        # 2 occurs 80% and 4 occurs 20%
        addNum = random.choice([2, 2, 2, 2, 2, 2, 2, 4, 4, 2])
        self.board[addIndex[0]][addIndex[1]] = addNum
        return addIndex, addNum

    def isSameBoard(self, board1, board2):
        if len(board1) != len(board2): return False
        elif len(board1[0]) != len(board2[0]): return False
        else:
            for i in range(len(board1)):
                for j in range(len(board1[0])):
                    if board1[i][j] != board2[i][j]: return False
            return True

    def moveLeft(self):
        check = copy.deepcopy(self.board)
        for row in range(self.size):
            if not list(filter (lambda x: x != self.empty, self.board[row])):
                pass # ignore empty rows
            else:
                # merge two same numbers, prior left 
                col = 0
                while col < self.size-1:
                    for nextCol in range(col+1, self.size):
                        if ((self.board[row][col] != self.empty) # a number
                            and (self.board[row][col] == self.board[row][nextCol])
                            # the tiles between the two same numbers are all empty
                            # or there're no tiles between them (i.e. two numbers are consecutive)
                            and (not list(filter (lambda x: x != self.empty, self.board[row][(col+1):nextCol])))): 
                            currNum = self.board[row][col]
                            self.board[row][col] = currNum * 2
                            self.score += self.board[row][col]
                            self.board[row][nextCol] = self.empty
                            col = nextCol
                            break
                    col += 1

                # move numbers to left all empty tiles on the left
                nums = list(filter (lambda x: x != self.empty, self.board[row]))
                newRow = nums + [0] * (self.size - len(nums))
                self.board[row] = newRow

        # return False if this move cannot be processed (i.e. the board after
        # moving is the same as the original one), else return True
        if self.isSameBoard(check, self.board): 
            return False
        return True
 
    # if we rotate the board 90˚ clockwise, move left and rotate 90° counterclockwise back, 
    # the new board is equivalent to the original board move down.

    # rotate the board 90° clockwise is equivalent to transpose the board and 
    # mirror the board along the vertical axis
    def transpose(self):
        board = copy.deepcopy(self.board)
        for i in range(self.size):
            for j in range(self.size):
                self.board[j][i] = board[i][j]

    def mirror(self):
        board = copy.deepcopy(self.board)
        for i in range(self.size):
            for j in range(self.size):
                self.board[i][j] = board[i][self.size-j-1]

    def rotate90Clockwise(self):
        self.transpose()
        self.mirror()

    def rotate180(self):
        self.transpose()
        self.mirror()
        self.transpose()
        self.mirror()

    def rotate90CounterClockwise(self):
        for i in range(3):
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

    def reaches2048(self): # have numbers greater than or equal to 2048
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] >= 2048: 
                    return True
        return False

    # the game is over once the player reach the number 2048 or 
    # cannot make any legal move on the board
    def GameOver(self):
        # board = copy.deepcopy(self.board)
        originalScore = self.score

        # the player get 2048
        if self.reaches2048():
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
            print("\nYou don't have any legal moves! Game Over!") 
            return True
        else: return False

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
                if canMove: self.addNewTile() 
                else: print("cannot move in this direction") 
            self.printBoard()
            if self.GameOver(): 
                print("Game Over!\n")
                break
        print("reach here after game is over!")


# test 
if __name__ == "__main__":
    board = Board(4)
    board.printBoard()
    board.addNewTile()
    board.printBoard()
    board.addNewTile()
    board.addNewTile()
    board.addNewTile()
    board.addNewTile()
    board.addNewTile()
    board.printBoard()
    print("after moving left:")
    board.moveLeft()
    board.printBoard()

    board1 = copy.deepcopy(board)
    print("-----------after transpose:")
    board1.transpose()
    board1.printBoard()
    print("after mirror:")
    board1.mirror()
    board1.printBoard()

    print("------------after rotation:")
    board.rotate90Clockwise()
    board.printBoard()

    print("------------rotate back:")
    board.rotate90CounterClockwise()
    board.printBoard()

    print("after moving down:")
    board.moveDown()
    board.printBoard()

    print("after moving up:")
    board.moveUp()
    board.printBoard()

    print("after moving right:")
    board.moveRight()
    board.printBoard()

    print("game over tests:")
    # test with infinite random moves
    board = Board(6)
    while not board.GameOver():
        canMove = False
        direction = random.choice(board.directionList)
        print("direction:", direction)
        if direction == "Up": 
            canMoveUp = board.moveUp()
            canMove = canMoveUp
        elif direction == "Down":
            canMoveDown = board.moveDown()
            canMove = canMoveDown
        elif direction == "Left":
            canMoveLeft = board.moveLeft()
            canMove = canMoveLeft
        elif direction == "Right":
            canMoveRight = board.moveRight()
            canMove = canMoveRight
        # add a new number after each legal move
        if canMove: board.addNewTile() 
        else: print("cannot move in this direction") 
        board.printBoard()
