## 2048 game AI

import random
import copy
import time
from board import Board


class AI:
    def __init__(self, GameBoard, level):
        self.size = GameBoard.size
        self.GameBoard = GameBoard
        self.level = [
            "novice",
            "advanced beginner",
            "competent",
            "proficient",
            "expert",
        ][level]

        '''
        Weight board assign the grids on board with weight
        in zigzag order increasing exponentially with base 4
        e.g. for a weight board with size 4, the board weight is
        [[4^0,   4^1,  4^2,   4^3],
         [4^7,   4^6,  4^5,   4^4],
         [4^8,   4^9,  4^10, 4^11],
         [4^15, 4^14,  4^13, 4^12]]

        Reference: http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf
        '''
        
        def weightBoard1(size):
            board = [[(row * size + col) for col in range(size)] for row in range(size)]
            for row in range(size):
                if row % 2:
                    board[row] = board[row][::-1]
            for row in range(size):
                for col in range(size):
                    exp = board[row][col]
                    board[row][col] = 4**exp
            return board

        self.weightBoard = weightBoard1(self.size)

    def nextMove(
        self,
    ):  # including both the optimal move of AI agent and the random move from computer
        if self.level == "novice":
            self.getMaxMove1()
        elif self.level == "advanced beginner":
            self.getMaxMove2()
        elif self.level == "competent":
            self.getMaxMove3()
        elif self.level == "proficient":
            self.getMaxMove4()
        elif self.level == "expert":
            self.getMaxMove5()

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
        if canMoveUp:
            legalMoves.append(0)

        canMoveDown = self.GameBoard.moveDown()
        self.GameBoard.score = originalScore
        self.GameBoard.board = boardD
        if canMoveDown:
            legalMoves.append(1)

        canMoveLeft = self.GameBoard.moveLeft()
        self.GameBoard.score = originalScore
        self.GameBoard.board = boardL
        if canMoveLeft:
            legalMoves.append(2)

        canMoveRight = self.GameBoard.moveRight()
        self.GameBoard.score = originalScore
        self.GameBoard.board = boardR
        if canMoveRight:
            legalMoves.append(3)

        return legalMoves

    def performAction(self, action):
        if action == 0:
            self.GameBoard.moveUp()
        elif action == 1:
            self.GameBoard.moveDown()
        elif action == 2:
            self.GameBoard.moveLeft()
        elif action == 3:
            self.GameBoard.moveRight()
        else:
            assert False  # should not reach here!
        # in reality, computer still generates numbers on board randomly
        self.GameBoard.addNewTile()

    # Novice AI: greedy search based on game board scores
    def getMaxMove1(self):
        actions = self.getLegalMoves()
        if not actions:
            return None  # no legal actions
        originalScore = self.GameBoard.score

        bestScore, bestActions = -float("inf"), []
        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            score = self.GameBoard.score - originalScore
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if score > bestScore:
                bestActions = [action]
                bestScore = score
            elif score == bestScore:
                bestActions.append(action)

        legalMoves = self.getLegalMoves()
        bestAction = (
            random.choice(bestActions) if bestActions else random.choice(legalMoves)
        )
        print("action: %d\n" % bestAction)
        self.performAction(bestAction)

    """
    The evaluate function estimates the current situation on the board and 
    return a score that quantifies the situation. The evaluation algorithm
    is that the score is equal to the sum of the product of weight of a 
    certain tile and the number on it. 
    (i.e. ∑(row)∑(col) weightBoard[row][col] * GameBoard[row][col])
    """

    def evaluate(self):
        score = 0
        for row in range(self.size):
            for col in range(self.size):
                score += self.weightBoard[row][col] * self.GameBoard.board[row][col]
        return score

    # Advanced Beginner AI: greedy search based on game board scores with weights
    def getMaxMove2(self):
        actions = self.getLegalMoves()
        if not actions:
            return None  # no legal actions
        originalScore = self.GameBoard.score

        bestScore, bestActions = -float("inf"), []
        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            score = self.evaluate()
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if score > bestScore:
                bestActions = [action]
                bestScore = score
            elif score == bestScore:
                bestActions.append(action)

        legalMoves = self.getLegalMoves()
        bestAction = (
            random.choice(bestActions) if bestActions else random.choice(legalMoves)
        )
        print("action: %d\n" % bestAction)
        self.performAction(bestAction)

    ### ExpectiMax & miniMax algorithm.
    # Reference: http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    """
    2021.12.22 update: we should consider the number of merges for each move and the 
    number of empty tiles after each move of the player to be as many as possible in favor of the next move
    Reference: https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22389702#22389702
    """

    """
    ## ExpectiMax

    # player's move
    def expectiMaxieMove(self, depth, importance):
        if not depth: return (self.evaluate(), None) # depth = 0

        # get all legal actions and preserve the board
        originalScore = self.GameBoard.score

        actions = self.getLegalMoves()

        if not actions: return (self.evaluate(), None) # no legal actions

        (bestScore, bestAction) = (-float('inf'), None)

        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            computerScore = self.expectiMinnieScore(depth, importance)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action

        return (bestScore, bestAction)

    # computer's move
    def expectiMinnieScore(self, depth, importance):
        if not depth: return (self.evaluate(), None) # depth = 0

        originalScore = self.GameBoard.score

        # even though the real computer will put the new numbers randomly,
        # we still assume that it can put 2 or 4 on any empty tile as it 
        # wishes to make the board harder for player to solve.

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

        if not actions: return self.evaluate() # no legal actions

        expectedScore = 0

        for action in actions:
            addNum = action[1]
            if addNum == 2: prob = 0.8
            elif addNum == 4: prob = 0.2
            else: assert(False) # should not reach here

            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.addNewNum(action) # perform computer's action
            (playerScore, _) = self.expectiMaxieMove(depth-1, importance)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            expectedScore += playerScore * prob

        return expectedScore

    ## Minimax with alpha-beta pruning
    # player's move with alpha-beta pruning
    def maxieMoveAlphaBeta(self, depth, alpha, beta):
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
            (computerScore, computerAction) = self.minnieMoveAlphaBeta(depth-1, alpha, beta)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action
                alpha = max(alpha, bestScore)
                if (alpha >= beta): break

        return (bestScore, bestAction)

    # computer's move with alpha-beta pruning
    def minnieMoveAlphaBeta(self, depth, alpha, beta):
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
            (playerScore, playerAction) = self.maxieMoveAlphaBeta(depth-1, alpha, beta)
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if playerScore < bestScore:
                bestScore = playerScore
                bestAction = action
                beta = min(beta, bestScore)
                if (alpha >= beta) : break

        return (bestScore, bestAction)
    """

    """
    Importance pruning: only take the computer's actions that affect the player's next move most negatively based on the weight of the empty tiles on the board.
    Reference : http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf

    2023.01.12 update: we consider only some most important empty tiles, where importance is proportional to the weight attached to a tile; according to the
    conclusion of the paper, to consider four or fewer empty tiles at each depth could better balance the need for both high score and running time.
    To be more specific, for maximum depth of 4, the search tree considers no more than 4 empty tiles at first layer; 
    then it considers no more than 3 empty tiles at one layer down; in the last but one layer it would consider at most one empty tile.
    """

    # the method for computer that does not add the numbers "normally" -- add the number based on the current game board situation
    def addNewNum(self, action):
        index = action[0]
        addNum = action[1]
        assert self.GameBoard.board[index[0]][index[1]] == 0
        self.GameBoard.board[index[0]][index[1]] = addNum

    # the method to get most important tiles for computer's actions
    def getImporantTiles(
        self, importance
    ):  # importance => number of important tiles we consider (in current layer)
        emptyTiles = self.GameBoard.findAllEmptyTiles()
        importantTiles = sorted(
            emptyTiles,
            key=lambda coord: self.weightBoard[coord[0]][coord[1]],
            reverse=True,
        )[
            :importance
        ]  # len(importantTiles) == importance
        return importantTiles

    # player's move with alpha-beta pruning & importance pruning
    def maxieMoveAlphaBetaImportance(self, depth, alpha, beta, importance, evalFunc):
        assert alpha < beta
        if not depth:
            return evalFunc(), None  # depth = 0

        # get all legal actions and preserve the board
        originalScore = self.GameBoard.score
        actions = self.getLegalMoves()
        if not actions:
            return (
                evalFunc(),
                None,
            )  # no legal actions, means player loses => computer wins

        bestScore, bestAction = float("-inf"), None
        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.performAction(action)
            computerScore, _ = self.minnieMoveAlphaBetaImportance(
                depth - 1, alpha, beta, importance - 1, evalFunc
            )
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if computerScore > bestScore:
                bestScore = computerScore
                bestAction = action
                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break

        return bestScore, bestAction

    # computer's move with alpha-beta pruning & importance pruning
    def minnieMoveAlphaBetaImportance(self, depth, alpha, beta, importance, evalFunc):
        assert alpha < beta
        if not depth:
            return evalFunc(), None  # depth = 0

        originalScore = self.GameBoard.score
        '''
        Even though the real computer will put the new numbers randomly,
        we still assume that it can put 2 or 4 on any empty tile as it
        wishes to make the board harder for player to solve.
        '''

        # mark the empty tiles with highest importances as "important"
        importantTiles = self.getImporantTiles(importance)
        actions = []
        for index in importantTiles:
            # can add 2 or 4 on any empty tile
            actions.append((index, 2))
            actions.append((index, 4))

        if not actions:
            return evalFunc(), None

        bestScore, bestAction = float("inf"), None
        for action in actions:
            beforeMoveBoard = copy.deepcopy(self.GameBoard.board)
            self.addNewNum(action)  # perform computer's action
            playerScore, _ = self.maxieMoveAlphaBetaImportance(
                depth, alpha, beta, importance, evalFunc
            )
            self.GameBoard.board = beforeMoveBoard
            self.GameBoard.score = originalScore

            if playerScore < bestScore:
                bestScore = playerScore
                bestAction = action
                beta = min(beta, bestScore)
                if alpha >= beta:
                    break

        return bestScore, bestAction

    def getMaxMove3(self):
        score, action = self.maxieMoveAlphaBetaImportance(
            4, -float("inf"), float("inf"), 4, self.evaluate
        )
        print(
            "bestScore: {score}, bestAction: {action}".format(
                score=score, action=self.GameBoard.directionList[action]
            )
        )
        self.performAction(action)

    """
    2023.09.02 Update: Second evaluation function for Expectiminimax that the score is equal to 
    the product of the current board score and the sum of the product of weight of a certain tile and the number on it. 
    (i.e. GameBoard.score * ∑(row)∑(col) weightBoard[row][col] * GameBoard[row][col])
    """
    def evaluate2(self):
        score = 0
        for row in range(self.size):
            for col in range(self.size):
                score += self.weightBoard[row][col] * self.GameBoard.board[row][col]
        return self.GameBoard.score * score

    def getMaxMove4(self):
        score, action = self.maxieMoveAlphaBetaImportance(
            4, -float("inf"), float("inf"), 4, self.evaluate2
        )
        print(
            "bestScore: {score}, bestAction: {action}".format(
                score=score, action=self.GameBoard.directionList[action]
            )
        )
        self.performAction(action)

    def getMaxMove5(self):
        # apply deep reinforcement learning
        print("有时间一定会做的")
        raise NotImplementedError

    # this function is used for terminal playing version
    def playTheGame(self):
        step = 0
        print("start board: ", end="")
        self.GameBoard.printBoard()
        while not self.GameBoard.gameOver():
            step += 1
            print("step:%d\n" % step)
            print("-------------------------------board before move:")
            self.GameBoard.printBoard()
            self.nextMove()
            print("\n-----------------------------board after move:")
            self.GameBoard.printBoard()
            print("current score: ", self.GameBoard.score)
            print("------------------------------------------------\n\n")
        return (
            int(self.GameBoard.reaches2048()),
            self.GameBoard.getLargestTileNumber(),
            self.GameBoard.score,
        )


# test
if __name__ == "__main__":
    # # novice AI plays 100 times
    # record = []
    # scores = []
    # for i in range(100):
    #     testBoard = Board(4)
    #     noviceAI = AI(testBoard, 0)
    #     res = noviceAI.playTheGame()
    #     record.append(res[0])
    #     scores.append(res[1])
    # print("Novice AI:")
    # print("record:", record)
    # print("scores:", scores)
    # avgscore = sum(scores)/len(scores)
    # print("average score: ", avgscore)
    # winrate = sum(record)/len(record)
    # print("winrate: ", winrate)

    # # advanced beginner AI plays 100 times
    # startTime = time.time()
    # winLose, record, scores = [], [], []
    # for i in range(100):
    #     testBoard = Board(4)
    #     competentAI = AI(testBoard, 1)
    #     res = competentAI.playTheGame()
    #     winLose.append(res[0])
    #     record.append(res[1])
    #     scores.append(res[2])
    # print("Advanced Beginner AI:")
    # print("winLose: ", winLose)
    # print("record:", record)
    # print("scores:", scores)
    # avgscore = sum(scores)/len(scores)
    # print("average score: ", avgscore)
    # winrate = sum(winLose)/len(record)
    # print("winrate: ", winrate)
    # print("--- %s seconds ---" % (time.time()-startTime))

    # competent AI plays 20 times
    startTime = time.time()
    winLose, record, scores = [], [], []
    for i in range(20):
        currTrialStartTime = time.time()
        testBoard = Board(4)
        competentAI = AI(testBoard, 2)
        res = competentAI.playTheGame()
        winLose.append(res[0])
        record.append(res[1])
        scores.append(res[2])
        print(
            "---Current trial time: %s seconds ---" % (time.time() - currTrialStartTime)
        )
    print("Competent AI:")
    print("winLose: ", winLose)
    print("record:", record)
    print("scores:", scores)
    avgscore = sum(scores) / len(scores)
    print("average score: ", avgscore)
    winrate = sum(winLose) / len(record)
    print("winrate: ", winrate)
    print("---Total time: %s seconds ---" % (time.time() - startTime))

    # # proficient AI play 20 times
    # startTime = time.time()
    # winLose, record, scores = [], [], []
    # for i in range(20):
    #     currTrialStartTime = time.time()
    #     testBoard = Board(4)
    #     proficientAI = AI(testBoard, 3)
    #     res = proficientAI.playTheGame()
    #     winLose.append(res[0])
    #     record.append(res[1])
    #     scores.append(res[2])
    #     print(
    #         "---Current trial time: %s seconds ---" % (time.time() - currTrialStartTime)
    #     )
    # print("Proficient AI:")
    # print("winLose: ", winLose)
    # print("record:", record)
    # print("scores:", scores)
    # avgscore = sum(scores) / len(scores)
    # print("average score: ", avgscore)
    # winrate = sum(winLose) / len(record)
    # print("winrate: ", winrate)
    # print("---Total time: %s seconds ---" % (time.time() - startTime))