## 2048 game board

import random
import copy

class Board(object):
    def __init__(self, size):
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

    def addNewTile(self, board):
        emptyTiles = [] # tuple list => empty tile coordinates
        for i in range(self.size):
            for j in range(self.size):
                if not board[i][j]: # this tile is empty
                    emptyTiles.append((i, j))
        addIndex = random.choice(emptyTiles)
        # 2 occurs 80% and 4 occurs 20%
        addNum = random.choice([2, 2, 2, 2, 2, 2, 2, 4, 4, 2])
        board[addIndex[0]][addIndex[1]] = addNum
        return board

    def isSameBoard(self, board1, board2):
        if len(board1) != len(board2): return False
        elif len(board1[0]) != len(board2[0]): return False
        else:
            for i in range(len(board1)):
                for j in range(len(board1[0])):
                    if board1[i][j] != board2[i][j]: return False
            return True

    def moveLeft(self, board):
        check = copy.deepcopy(board)
        res = copy.deepcopy(board)
        for row in range(self.size):
            if not list(filter (lambda x: x != self.empty, self.board[row])):
                pass # ignore empty rows
            else:
                # merge two same numbers, prior left 
                col = 0
                while col < self.size-1:
                    for nextCol in range(col+1, self.size):
                        if ((res[row][col] != self.empty) # a number
                            and (res[row][col] == res[row][nextCol])
                            # the tiles between the two same numbers are all empty
                            # or there're no tiles between them (i.e. two numbers are consecutive)
                            and (not list(filter (lambda x: x != self.empty, res[row][(col+1):nextCol])))): 
                            currNum = res[row][col]
                            res[row][col] = currNum * 2
                            self.score += res[row][col]
                            res[row][nextCol] = self.empty
                            col = nextCol
                            break
                    col += 1

                # move numbers to left all empty tiles on the left
                nums = list(filter (lambda x: x != self.empty, res[row]))
                newRow = nums + [0] * (self.size - len(nums))
                res[row] = newRow

        # return False if this move cannot be processed (i.e. the board after
        # moving is the same as the original one), else return True
        if self.isSameBoard(check, res): 
            return (False, res)
        return (True, res)
 
    # if we rotate the board 90˚ clockwise, move left and rotate 90° counterclockwise back, 
    # the new board is equivalent to the original board move down.

    # rotate the board 90° clockwise is equivalent to transpose the board and 
    # mirror the board along the vertical axis
    def transpose(self, board):
        tempBoard = copy.deepcopy(board)
        for i in range(self.size):
            for j in range(self.size):
                board[j][i] = tempBoard[i][j]
        return board

    def mirror(self, board):
        tempBoard = copy.deepcopy(board)
        for i in range(self.size):
            for j in range(self.size):
                board[i][j] = tempBoard[i][self.size-j-1]
        return board

    def rotate90Clockwise(self, board):
        board = self.transpose(board)
        board = self.mirror(board)
        return board

    def rotate180(self, board):
        board = self.transpose(board)
        board = self.mirror(board)
        board = self.transpose(board)
        board = self.mirror(board)
        return board

    def rotate90CounterClockwise(self, board):
        for i in range(3):
            board = self.transpose(board)
            board = self.mirror(board)
        return board

    def moveDown(self, board):
        board = self.rotate90Clockwise(board)
        (canMove, board) = self.moveLeft(board)
        board = self.rotate90CounterClockwise(board)
        return (canMove, board)

    def moveUp(self, board):
        board = self.rotate90CounterClockwise(board)
        (canMove, board) = self.moveLeft(board)
        board = self.rotate90Clockwise(board)
        return (canMove, board)

    def moveRight(self, board):
        board = self.rotate180(board)
        (canMove, board) = self.moveLeft(board)
        board = self.rotate180(board)
        return (canMove, board)

    def contains2048(self, board):
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 2048: 
                    return True
        return False

    # the game is over once the player reach the number 2048 or 
    # cannot make any legal move on the board
    def GameOver(self, board):
        originalScore = self.score
        originalBoard = copy.deepcopy(board)
        # the player get 2048
        if self.contains2048(board):
            print("\nCongratulations! you get 2048 and win!\n")
            return True

        # the player cannot make any legal move before getting 2048
        ((canMoveUp, _), (canMoveDown, _), (canMoveLeft, _), (canMoveRight, _)) \
            = (self.moveUp(board), self.moveDown(board), self.moveLeft(board), self.moveRight(board))
        if not (canMoveUp or canMoveDown or canMoveLeft or canMoveRight):
            print("\nYou don't have any legal moves! Game Over!") 
            return True
        else: 
            self.score = originalScore
            self.board = originalBoard
            return False



    
#     # this function is used for terminal playing version
#     def move(self):
#         while 1:
#             direction = input("moving direction (Up, Down, Left, Right): ")
#             canMove = False
#             if direction not in self.directionList: 
#                 print("illegal move")
#             else:
#                 print("direction:", direction)
#                 if direction == "Up": 
#                     canMoveUp = self.moveUp()
#                     canMove = canMoveUp
#                 elif direction == "Down":
#                     canMoveDown = self.moveDown()
#                     canMove = canMoveDown
#                 elif direction == "Left":
#                     canMoveLeft = self.moveLeft()
#                     canMove = canMoveLeft
#                 elif direction == "Right":
#                     canMoveRight = self.moveRight()
#                     canMove = canMoveRight
#                 # add a new number after each legal move
#                 if canMove: self.addNewTile() 
#                 else: print("cannot move in this direction") 
#             self.printBoard()
#             if self.GameOver(): 
#                 print("Game Over!\n")
#                 break
#         print("reach here after game is over!")


# test 
if __name__ == "__main__":
    gameBoard = Board(4)
    gameBoard.printBoard()
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.printBoard()
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.board = gameBoard.addNewTile(gameBoard.board)
    gameBoard.printBoard()
    print("after moving left:")
    gameBoard.board = gameBoard.moveLeft(gameBoard.board)[1]
    gameBoard.printBoard()

    # board1 = copy.deepcopy(board)
    # print("-----------after transpose:")
    # board1.transpose()
    # board1.printBoard()
    # print("after mirror:")
    # board1.mirror()
    # board1.printBoard()

    # print("------------after rotation:")
    # board.rotate90Clockwise()
    # board.printBoard()

    # print("------------rotate back:")
    # board.rotate90CounterClockwise()
    # board.printBoard()

    print("after moving down:")
    gameBoard.board = gameBoard.moveDown(gameBoard.board)[1]
    gameBoard.printBoard()

    print("after moving Up:")
    gameBoard.board = gameBoard.moveUp(gameBoard.board)[1]
    gameBoard.printBoard()

    print("after moving right:")
    gameBoard.board = gameBoard.moveRight(gameBoard.board)[1]
    gameBoard.printBoard()

    print("game over tests:")
    # test with infinite random moves
    gameBoard = Board(6)
    while not gameBoard.GameOver(gameBoard.board):
        canMove = False
        direction = random.choice(gameBoard.directionList)
        print("direction:", direction)
        if direction == "Up": 
            canMoveUp, board = gameBoard.moveUp(gameBoard.board)
            gameBoard.board = board
            canMove = canMoveUp
        elif direction == "Down":
            canMoveDown, board = gameBoard.moveDown(gameBoard.board)
            gameBoard.board = board
            canMove = canMoveDown
        elif direction == "Left":
            canMoveLeft, board = gameBoard.moveLeft(gameBoard.board)
            gameBoard.board = board
            canMove = canMoveLeft
        elif direction == "Right":
            canMoveRight, board = gameBoard.moveRight(gameBoard.board)
            gameBoard.board = board
            canMove = canMoveRight
        # add a new number after each legal move
        if canMove: gameBoard.board = gameBoard.addNewTile(gameBoard.board)
        else: print("cannot move in this direction") 
        gameBoard.printBoard()

#     print("----------------------------------------------------------------")
#     print("move with commands:")
#     board = Board(4)
#     board.move()


