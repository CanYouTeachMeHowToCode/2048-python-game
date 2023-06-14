## Deep Reinforcement Learning part

import random
import copy
import math

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np

from board import Board

# Reference: https://github.com/navjindervirdee/2048-deep-reinforcement-learning
# for deep reinforcement learning: currently only consider board with size 4 (can be improved later)

# # convert the input game board (or batch of game boards) into corresponding power of 2 tensor.
# def convertBoard2PowBoard(board):
#     batchSize = board.size()[0]
#     powerBoard = np.zeros(shape=(batchSize, 16, board.size()[1], board.size()[2]), dtype=np.float32) # 16 is for board size 4 (2^0 to 2^15)
#     for batchIndex in range(batchSize):
#         for i in range(board.size()[1]):
#             for j in range(board.size()[2]):
#                 if not board[batchIndex][i][j]:
#                     powerBoard[batchIndex][0][i][j] = 1.0
#                 else:
#                     power = int(math.log(board[batchIndex][i][j],2))
#                     powerBoard[batchIndex][power][i][j] = 1.0
#     return torch.from_numpy(powerBoard).to(device) # NCHW format, different from NHWC in tf

# find the number of empty tiles in the game board.
def findEmptyTiles(board):
    count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if not board[i][j]:
                count += 1
    return count

# if gpu is to be used
print("torch.cuda.is_available(): ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network Architecture
batchSize = 512
inputSize = 16 # board size
hiddenSize = 256
outputSize = 4 # number of moves

class MyPolicy(nn.Module):
    def __init__(self):
        super(MyPolicy, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.fc1.weight.data.normal_(0, 0.01) # initialization
        self.out = nn.Linear(hiddenSize, outputSize)
        self.out.weight.data.normal_(0, 0.01) # initialization

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x)
        return out

# helper function
def save_policy(policyNet):
    torch.save(policyNet.state_dict(), "mypolicy.pth")

# def initWeightBias(layer):
#     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#         nn.init.kaiming_uniform_(layer.weight, mode='fan_in')
#         nn.init.zeros_(layer.bias)

policyNet = MyPolicy()
# policyNet.load_state_dict(torch.load("mypolicy.pth"))
print("policyNet", policyNet)

# # unit test
# GameBoard = Board(4)
# GameBoard.printBoard()
# print("asd", torch.from_numpy(np.expand_dims(GameBoard.board, axis=0)))
# print(convertBoard2PowBoard(torch.from_numpy(np.expand_dims(GameBoard.board, axis=0))))
# GameBoard.moveLeft()
# GameBoard.printBoard()
# print(convertBoard2PowBoard(torch.from_numpy(np.expand_dims(GameBoard.board, axis=0))))
# GameBoard.addNewTile()
# GameBoard.printBoard()
# print(convertBoard2PowBoard(torch.from_numpy(np.expand_dims(GameBoard.board, axis=0))))
# print("GameBoard.board", GameBoard.board)

# GameBoard2 = Board(4)
# GameBoard2.printBoard()
# GameBoard2.moveLeft()
# GameBoard2.printBoard()
# GameBoard2.addNewTile()
# GameBoard2.printBoard()
# GameBoard2.moveUp()
# GameBoard2.printBoard()
# GameBoard2.addNewTile()
# GameBoard2.printBoard()

# print("GameBoard2.board", GameBoard2.board)
# print("np.stack(GameBoard.board, GameBoard2.board)", np.stack((GameBoard.board, GameBoard2.board)))
# testInput = convertBoard2PowBoard(torch.from_numpy(np.stack((GameBoard.board, GameBoard2.board))))
# res = policyNet(testInput)
# print("res", res)
# print("torch.max(res, 1)[1].data.numpy()", torch.max(res, 1)[1].data.numpy())

# Hyper Parameters
LR = 5e-3 # learning rate
GAMMA = 0.9 # reward discount
EPSILON = 0.95 # for epsilon greedy algorithm
# REPLAY_MEMORY, REPLAY_LABELS = [], []

BATCH_SIZE = 512
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 6000

num_episodes = 100000

# for episode with max score
max_episode = -1
max_score = -1

# total scores
scores = []

class DQN:
    def __init__(self):
        self.eval_net, self.target_net = policyNet.to(device), policyNet.to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, inputSize*2+2))        # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def chooseAction(self, GameBoard):
        inputState = torch.from_numpy(np.expand_dims(np.array(GameBoard.board).flatten(), axis=0)).type('torch.FloatTensor')
        # convertBoard2PowBoard(torch.from_numpy(np.expand_dims(GameBoard.board, axis=0)))
        
        # input only one sample
        legalMoves = getLegalMoves(GameBoard)
        if np.random.uniform() <= EPSILON:   # greedy
            actionsQvalue = self.eval_net(inputState)
            # action should first be lgeal actions
            illegalMask = list(set([0, 1, 2, 3]).difference(set(legalMoves)))
            for i in illegalMask: actionsQvalue[0][i] = float('-inf')
            # print("actionsQvalue: ", actionsQvalue)
            action = torch.max(actionsQvalue, 1)[1].cpu().data.numpy()[0] # action with maximum Q value
            # action = # action should be legal
        else:   # random
            action = random.choice(legalMoves)
        return action

    def store_transition(self, GameBoard, action, reward, newGameBoard):
        transition = np.hstack((np.array(GameBoard.board).flatten(), [action, reward], np.array(newGameBoard.board).flatten()))
        # print("transition: ", transition)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :inputSize]).to(device)
        b_a = torch.LongTensor(b_memory[:, inputSize:inputSize+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, inputSize+1:inputSize+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -inputSize:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print("loss", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

def getLegalMoves(GameBoard):
    originalScore = GameBoard.score
    boardU = copy.deepcopy(GameBoard.board)
    boardD = copy.deepcopy(GameBoard.board)
    boardL = copy.deepcopy(GameBoard.board)
    boardR = copy.deepcopy(GameBoard.board)

    # legal actions for player is represented by numbers:
    # 0 : Up
    # 1 : Down
    # 2 : Left
    # 3 : Right
    legalMoves = []

    canMoveUp = GameBoard.moveUp()
    GameBoard.score = originalScore      
    GameBoard.board = boardU
    if canMoveUp : legalMoves.append(0)

    canMoveDown = GameBoard.moveDown()
    GameBoard.score = originalScore    
    GameBoard.board = boardD
    if canMoveDown : legalMoves.append(1)

    canMoveLeft = GameBoard.moveLeft()
    GameBoard.score = originalScore      
    GameBoard.board = boardL
    if canMoveLeft : legalMoves.append(2)

    canMoveRight = GameBoard.moveRight()
    GameBoard.score = originalScore
    GameBoard.board = boardR
    if canMoveRight : legalMoves.append(3)

    return legalMoves

def getWeightBoard(size):
    board = [[(row * size + col) for col in range(size)] for row in range(size)]
    for row in range(size):
        if row % 2 : board[row] = board[row][::-1]
    for row in range(size):
        for col in range(size):
            exp = board[row][col]
            board[row][col] = 4 ** exp
    return board

def evaluateBoard(board, weightBoard):
    score = 0
    for row in range(len(board)):
        for col in range(len(board)):
            score += (board[row][col] * weightBoard[row][col])
    return score

def step(GameBoard, action):
    originalBoard = copy.deepcopy(GameBoard.board)
    originalScore = GameBoard.score
    if action == 0: GameBoard.moveUp()
    elif action == 1: GameBoard.moveDown()
    elif action == 2: GameBoard.moveLeft()
    elif action == 3: GameBoard.moveRight()
    else: assert(False) # should not reach here!

    newBoard = GameBoard.board
    # Current Reward = number of merges + log2(weighted board total score)
    mergeNum = findEmptyTiles(newBoard)-findEmptyTiles(originalBoard)
    if not mergeNum: mergeNum = -math.log2(np.max(newBoard)) # penalty
    GameBoard.addNewTile()
    newBoard = GameBoard.board
    weightBoard = getWeightBoard(len(newBoard))
    reward = mergeNum+math.log2(evaluateBoard(newBoard, weightBoard))
    if GameBoard.GameOver(): done = True
    else: done = False
    moveScore = GameBoard.score - originalScore
    return GameBoard, reward, done, moveScore

# # unit test on step()
# GameBoard = Board(4)
# GameBoard.printBoard()
# GameBoard.moveLeft()
# GameBoard.addNewTile()
# GameBoard.printBoard()
# GameBoard.moveRight()
# GameBoard.addNewTile()
# GameBoard.printBoard()
# GameBoard.moveUp()
# GameBoard.addNewTile()
# GameBoard.printBoard()
# GameBoard.moveRight()
# GameBoard.addNewTile()
# GameBoard.printBoard()
# GameBoard, reward, done, moveScore = step(GameBoard, 2) # move left
# GameBoard.printBoard()
# print("reward: ", reward)
# print("done: ", done)
# print("moveScore: ", moveScore)
# assert(False)

## training 
print('\nCollecting experience...')
for episode in range(num_episodes):
    # start state
    GameBoard = Board(4)
    GameBoard.addNewTile()
    episode_reward = 0

    # whether current episode finished or not
    done = False
    
    # total score of this episode
    totalScore = 0
    
    while(1):
        # choose action
        action = dqn.chooseAction(GameBoard)

        # take action
        newGameBoard, reward, done, moveScore = step(GameBoard, action)

        # score of the corresponding action
        totalScore += moveScore
        
        # # decrease the epsilon value during training
        # if (episode > 10000) or (EPSILON > 0.1 and total_iters%2500==0):
        #     EPSILON = EPSILON/1.005
       
        # store them in memory
        dqn.store_transition(GameBoard, action, reward, newGameBoard)

        episode_reward += reward

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('episode: ', episode,
                      '| episode_reward: ', round(episode_reward, 2))

        if done: break

        GameBoard = newGameBoard
        
    scores.append(totalScore)
    print("Episode {} finished with score {}, epsilon : {}, learning rate : {} ".format(episode, totalScore, EPSILON, LR))
    GameBoard.printBoard()
    
    if not (episode+1) % 1000:
        print("Maximum Score : {} ,Episode : {}".format(max_score, max_episode))    
        print("\n")
        
    if max_score < totalScore:
        max_score = totalScore
        max_episode = episode

print("Scores: ", scores)
print("Maximum Score : {} ,Episode : {}".format(max_score, max_episode))    

def save_policy(policyNet):
    torch.save(policyNet.state_dict(), "mypolicy.pth")

save_policy(policyNet)



