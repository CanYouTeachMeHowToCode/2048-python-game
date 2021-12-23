## 2048 game AI

import random
import copy
import math
from board import Board

class AI(object):
    def __init__(self, GameBoard, level):
        self.size = GameBoard.size
        self.GameBoard = GameBoard
        self.level = ["easy", "normal", "hard"][level]

        # weight board assign the grids on board with weight 
        # in zigzag order increasing exponentially with base 4
        # e.g. for a weight board with size 4, the board weight is 
            # [[4^0,   4^1,  4^2,   4^3],
            #  [4^7,   4^6,  4^5,   4^4],
            #  [4^8,   4^9,  4^10, 4^11],
            #  [4^15, 4^14,  4^13, 4^12]]
        # Reference: http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
        def weightBoard1(size):
            board = [[(row * size + col) for col in range(size)] for row in range(size)]
            for row in range(size):
                if row % 2 : board[row] = board[row][::-1]
            for row in range(size):
                for col in range(size):
                    exp = board[row][col]
                    board[row][col] = 4 ** exp
            return board

        # another weightBoard assign the grids on baord with weight that force 
        # the grid to reach monotonicity
        # e.g. for a weight board with size 4, the board weight is 
            # [[4^3,   4^2,  4^1,   4^0],
            #  [4^4,   4^3,  4^2,   4^1],
            #  [4^5,   4^4,  4^3,   4^2],
            #  [4^6,   4^5,  4^4,   4^3]]
        # Reference: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22389702#22389702
        def weightBoard2(size):
            board = [[2*(row + col) for col in range(size)] for row in range(size)]
            for row in range(size):
                board[row] = board[row][::-1]
            for row in range(size):
                for col in range(size):
                    exp = board[row][col]
                    board[row][col] = 4 ** exp
            return board

        self.weightBoard = weightBoard1(self.size)
        print("self.weightBoard\n", self.weightBoard)

    def nextMove(self):
        if self.level == "easy": self.getMaxMove1()
        elif self.level == "normal": self.getMaxMove2()
        elif self.level == "hard" : self.getMaxMove3()
        else: pass
            # self.getMaxMove4() 

    def getLegalMoves(self):
        originalScore = self.GameBoard.score
        boardU = copy.deepcopy(self.GameBoard.board)
        boardD = copy.deepcopy(self.GameBoard.board)
        boardL = copy.deepcopy(self.GameBoard.board)
        boardR = copy.deepcopy(self.GameBoard.board)

        # legal actions for player is represented by numbers:
        # 0 : Up
        # 1 : Down
        # 2 : Left
        # 3 : Right
        legalMoves = []

        canMoveUp = self.GameBoard.moveUp()
        self.GameBoard.score = originalScore      
        self.GameBoard.board = boardU
        if canMoveUp : legalMoves.append(0)

        canMoveDown = self.GameBoard.moveDown()
        self.GameBoard.score = originalScore    
        self.GameBoard.board = boardD
        if canMoveDown : legalMoves.append(1)

        canMoveLeft = self.GameBoard.moveLeft()
        self.GameBoard.score = originalScore      
        self.GameBoard.board = boardL
        if canMoveLeft : legalMoves.append(2)

        canMoveRight = self.GameBoard.moveRight()
        self.GameBoard.score = originalScore
        self.GameBoard.board = boardR
        if canMoveRight : legalMoves.append(3)

        return legalMoves

    def performAction(self, action):
        if action == 0: self.GameBoard.moveUp()
        elif action == 1: self.GameBoard.moveDown()
        elif action == 2: self.GameBoard.moveLeft()
        elif action == 3: self.GameBoard.moveRight()
        else: assert(False) # should not reach here!

    # using simple algorithm that only counts the current step that can reach 
    # the highest score
    def getMaxMove1(self):
        assert(not self.GameBoard.GameOver())
        # maxie move
        upScore = downScore = leftScore = rightScore = -1
        originalScore = self.GameBoard.score
        boardU = copy.deepcopy(self.GameBoard.board)
        boardD = copy.deepcopy(self.GameBoard.board)
        boardL = copy.deepcopy(self.GameBoard.board)
        boardR = copy.deepcopy(self.GameBoard.board)
        canMoveUp = self.GameBoard.moveUp()
        if canMoveUp: upScore = self.GameBoard.score - originalScore
        self.GameBoard.board = boardU
        self.GameBoard.score = originalScore

        canMoveDown = self.GameBoard.moveDown()
        if canMoveUp: downScore = self.GameBoard.score - originalScore
        self.GameBoard.board = boardD
        self.GameBoard.score = originalScore

        canMoveLeft = self.GameBoard.moveLeft()
        if canMoveLeft: leftScore = self.GameBoard.score - originalScore
        self.GameBoard.board = boardL
        self.GameBoard.score = originalScore

        canMoveRight = self.GameBoard.moveRight()
        if canMoveRight: rightScore = self.GameBoard.score - originalScore
        self.GameBoard.board = boardR
        self.GameBoard.score = originalScore

        scoreList = [upScore, downScore, leftScore, rightScore]
        # moving randomly when each direction has the same score
        if (len(set(scoreList)) <= 1) : action = random.randint(0, 3)
        else: action = scoreList.index(max(scoreList))
        print("action: %d\n" % action)
        self.performAction(action)

        # (fake computer minie move that just uses the normal method to add new numbers)
        self.GameBoard.addNewTile()

    # ExpectiMax & miniMax algorithm. 
    # Reference: http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    # the evaluate function estimates the current situation on the board and 
    # return a score that quantifies the situation. The evaluation algorithm
    # is that the score is equal to the sum of the product of weight of a 
    # certain tile and the number on it. 
    # (i.e. ∑(row)∑(col) weightBoard[row][col] * GameBoard[row][col])

    # 2021.12.22 update: we should consider the number of merges for each move and the 
    # number of empty tiles after each move of the player to be as many as possible in favor of the next move
    # Reference: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22389702#22389702
    def evaluate(self):
        score = 0
        emptyTilesNum = 0
        for row in range(self.size):
            for col in range(self.size):
                if not self.GameBoard.board[row][col]: emptyTilesNum += 1
                score += (self.weightBoard[row][col] * self.GameBoard.board[row][col])
        # score += math.log(1+emptyTilesNum)*3
        score += emptyTilesNum
        return score

    # the method for computer that does not add the numbers "normally" --
    # add the number based on the current game board situation
    def addNewNum(self, action):  
        index = action[0]
        addNum = action[1]
        assert(self.GameBoard.board[index[0]][index[1]] == 0)
        self.GameBoard.board[index[0]][index[1]] = addNum

    ## ExpectiMax
    # player's move
    def expectiMaxieMove(self, depth):
        if not depth: return (self.evaluate(), None) # depth = 0

        # get all legal actions and preserve the board
        originalScore = self.GameBoard.score

        actions = self.getLegalMoves()

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (-float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            computerScore = self.expectiMinnieScore(depth)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action

        return (bestScore, bestAction)

    # computer's move
    def expectiMinnieScore(self, depth):
        if not depth: return (self.evaluate(), None) # depth = 0

        originalScore = self.GameBoard.score

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

        if not actions: return (self.evaluate(), None) # no legal actions

        # (bestScore, bestAction) = (float('inf'), None)
        expectedScore = 0

        for action in actions:
            addNum = action[1]
            if addNum == 2: prob = 0.8
            elif addNum == 4: prob = 0.2
            else: assert(False) # should not reach here

            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.addNewNum(action) # perform computer's action
            (playerScore, _) = self.expectiMaxieMove(depth-1)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            # if playerScore < bestScore:
            #     bestScore = playerScore
            #     bestAction = action
            expectedScore += playerScore * prob

        return expectedScore

    ## Minimax 
    # player's move with alpha-beta pruning
    def expectiMaxieMoveAlphaBeta(self, depth, alpha, beta):
        assert(alpha < beta)
        if not depth: return (self.evaluate(), None) # depth = 0

        # get all legal actions and preserve the board
        originalScore = self.GameBoard.score
        actions = self.getLegalMoves()

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (-float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            (computerScore, computerAction) = self.expectiMinnieMoveAlphaBeta(depth-1, alpha, beta)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action
                alpha = max(alpha, bestScore)
                if (alpha >= beta): break

        return (bestScore, bestAction)

    # computer's move with alpha-beta pruning
    def expectiMinnieMoveAlphaBeta(self, depth, alpha, beta):
        assert(alpha < beta)
        if not depth: return (self.evaluate(), None) # depth = 0

        originalScore = self.GameBoard.score
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

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.addNewNum(action) # perform computer's action
            (playerScore, playerAction) = self.expectiMaxieMoveAlphaBeta(depth-1, alpha, beta)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if playerScore < bestScore:
                bestScore = playerScore
                bestAction = action
                beta = min(beta, bestScore)
                if (alpha >= beta) : break

        return (bestScore, bestAction)

    # importance pruning: only take the computer's actions that affect the player's next 
    # move most negatively based on the weight of the empty tiles on the board. 
    # Reference : http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    # the function to get these tiles with the greatest importance for computer's actions
    def getImporantIndices(self, importance):
        importantIndices = []
        board = self.weightBoard
        size = self.size
        importance_cutoff = size**2 - importance
        row_cutoff, col_cutoff = importance_cutoff//size, importance_cutoff%size
        for i in range(size):
            for j in range(size):
                if board[i][j] >= board[row_cutoff][col_cutoff]: importantIndices.append((i, j))
        return importantIndices

    # player's move with alpha-beta pruning & importance pruning
    def expectiMaxieMoveAlphaBetaImportance(self, depth, alpha, beta, importance):
        assert(alpha < beta)
        if not depth: return (self.evaluate(), None) # depth = 0

        # get all legal actions and preserve the board
        originalScore = self.GameBoard.score
        actions = self.getLegalMoves()

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (-float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            (computerScore, computerAction) = self.expectiMinnieMoveAlphaBetaImportance(depth-1, alpha, beta, importance)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action
                alpha = max(alpha, bestScore)
                if (alpha >= beta): break

        return (bestScore, bestAction)

    # computer's move with alpha-beta pruning & importance pruning
    def expectiMinnieMoveAlphaBetaImportance(self, depth, alpha, beta, importance):
        assert(alpha < beta)
        if not depth: return (self.evaluate(), None) # depth = 0

        originalScore = self.GameBoard.score
        # even though the real computer will put the new numbers randomly,
        # we still assume that it can put 2 or 4 on any empty tile as it 
        # wishes to make the board harder for player to solve.

        # mark the empty tiles with highest importances as "important"
        importantIndices = self.getImporantIndices(importance)
        emptyTiles = [] # tuple list => empty tile coordinates
        for i in range(self.size):
            for j in range(self.size):
                # this tile is empty and is "important"
                if not self.GameBoard.board[i][j] and (i, j) in importantIndices: 
                    emptyTiles.append((i, j))
        actions = []
        for index in emptyTiles:
            # can add 2 or 4 on any empty tile
            actions.append((index, 2))
            actions.append((index, 4))

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.addNewNum(action) # perform computer's action
            (playerScore, playerAction) = self.expectiMaxieMoveAlphaBetaImportance(depth-1, alpha, beta, importance)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if playerScore < bestScore:
                bestScore = playerScore
                bestAction = action
                beta = min(beta, bestScore)
                if (alpha >= beta) : break

        return (bestScore, bestAction)

    def getMaxMove2(self):
        (score, action) = self.expectiMaxieMove(2)
        # (score, action) = self.expectiMaxieMoveAlphaBetaImportance(5, -float('inf'), float('inf'), 8)
        # (score, action) = self.expectiMaxieMoveAlphaBeta(5, -float('inf'), float('inf'))

        # print("bestScore, bestAction:", (score, action))
        self.performAction(action)

        # in reality, computer still generates numbers on board randomly
        self.GameBoard.addNewTile()

    def getMaxMove3(self):
        # apply deep reinforcement learning
        print("有时间一定会做的")
        raise NotImplementedError

    # this function is used for terminal playing version
    def playTheGame(self):
        step = 0
        print("start board: ", end = "")
        self.GameBoard.printBoard()
        while not self.GameBoard.GameOver():
            print("-------------------------------board before move:")
            self.GameBoard.printBoard()
            step += 1
            print("step:%d\n" % step)
            self.nextMove()
            print("\n----------board after move----------------------:")
            self.GameBoard.printBoard()
            print("current score: ", self.GameBoard.score)
            print("------------------------------------------------\n\n")
        if self.GameBoard.contains2048(): return 1
        else: return 0

# test
if __name__ == "__main__":
    testBoard = Board(4)
    # easyAI = AI(4, 0)
    # easyAI.playTheGame()

    # play ten times
    record = []
    for i in range(30):
        testBoard = Board(4)
        normalAI = AI(testBoard, 1)
        record.append(normalAI.playTheGame())
        print("record:", record)
    winrate = sum(record)/len(record)
    print("winrate: ", winrate)

    # hardAI = AI(4, 2)
    # hardAI.playTheGame()


