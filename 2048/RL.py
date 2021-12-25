## Deep Reinforcement Learning part

import random
import copy
import math

import torch
from torch import optim
from torch import nn
import numpy as np

from board import Board

# Reference: https://github.com/navjindervirdee/2048-deep-reinforcement-learning
# for deep reinforcement learning: currently only consider board with size 4 (can be improved later)

# convert the input game board (or batch of game boards) into corresponding power of 2 tensor.
def convertBoard2PowBoard(board):
    batchSize = board.size()[0]
    powerBoard = np.zeros(shape=(batchSize, 16, board.size()[1], board.size()[2]), dtype=np.float32) # 16 is for board size 4 (2^0 to 2^15)
    for batchIndex in range(batchSize):
        for i in range(board.size()[1]):
            for j in range(board.size()[2]):
                if not board[batchIndex][i][j]:
                    powerBoard[batchIndex][0][i][j] = 1.0
                else:
                    power = int(math.log(board[batchIndex][i][j],2))
                    powerBoard[batchIndex][power][i][j] = 1.0
    return torch.from_numpy(powerBoard) # NCHW format, different from NHWC in tf

# find the number of empty tiles in the game board.
def findEmptyTiles(board):
    count = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if not board[i][j]:
                count += 1
    return count

# Network Architecture
depth1 = 128
depth2 = 128
batchSize = 512
inputSize = 16 # board size
hiddenSize = 256
outputSize = 4 # number of moves

class MyPolicy(nn.Module):
    def __init__(self):
        super(MyPolicy, self).__init__()
        self.Conv1A = nn.Sequential(         
            nn.Conv2d(inputSize, depth1, kernel_size=(1, 2)),                              
            nn.ReLU(),                        
        )

        self.Conv1B = nn.Sequential(         
            nn.Conv2d(inputSize, depth1, kernel_size=(2, 1)),                              
            nn.ReLU(),                     
        )

        self.Conv2A = nn.Sequential(         
            nn.Conv2d(depth1, depth2, kernel_size=(1, 2)),                              
            nn.ReLU(),                        
        )

        self.Conv2B = nn.Sequential(         
            nn.Conv2d(depth1, depth2, kernel_size=(2, 1)),                              
            nn.ReLU(),                        
        )

        self.L1 = nn.Sequential(
            nn.Linear(4*3*depth1*2+2*4*depth2*2+3*3*depth2*2, hiddenSize),
            nn.ReLU(),
        )

        self.out = nn.Sequential(         
            nn.Linear(hiddenSize, outputSize),  
            # nn.Softmax(0),                                                     
        )

    def forward(self, x):
        # Convolutional layers
        convA = self.Conv1A(x)
        convB = self.Conv1B(x)
        convAA = self.Conv2A(convA)
        convAB = self.Conv2B(convA)
        convBA = self.Conv2A(convB)
        convBB = self.Conv2B(convB)

        # Fully connected layers
        hiddenA = torch.reshape(convA, (convA.size()[0], convA.size()[1]*convA.size()[2]*convA.size()[3]))
        hiddenB = torch.reshape(convB, (convB.size()[0], convB.size()[1]*convB.size()[2]*convB.size()[3]))
        hiddenAA = torch.reshape(convAA, (convAA.size()[0], convAA.size()[1]*convAA.size()[2]*convAA.size()[3]))
        hiddenAB = torch.reshape(convAB, (convAB.size()[0], convAB.size()[1]*convAB.size()[2]*convAB.size()[3]))
        hiddenBA = torch.reshape(convBA, (convBA.size()[0], convBA.size()[1]*convBA.size()[2]*convBA.size()[3]))
        hiddenBB = torch.reshape(convBB, (convBB.size()[0], convBB.size()[1]*convBB.size()[2]*convBB.size()[3]))
        hidden = torch.cat([hiddenA, hiddenB, hiddenAA, hiddenAB, hiddenBA, hiddenBB], dim=1)
        hidden = self.L1(hidden)
        out = self.out(hidden)
        return out

# helper function
def save_policy(policyNet):
    torch.save(policyNet.state_dict(), "mypolicy.pth")

def initWeightBias(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in')
        nn.init.zeros_(layer.bias)

policyNet = MyPolicy()
policyNet.apply(initWeightBias)
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
LR = 5e-4 # learning rate
GAMMA = 0.9 # reward discount
EPSILON = 0.9 # for epsilon greedy algorithm
# REPLAY_MEMORY, REPLAY_LABELS = [], []

BATCH_SIZE = 512
TARGET_REPLACE_ITER = 100   # target update frequenc
MEMORY_CAPACITY = 6000

num_episodes = 20000

# for episode with max score
max_episode = -1
max_score = -1

# total iterations 
total_iters = 1

# total scores
scores = []

class DQN:
    def __init__(self):
        self.eval_net, self.target_net = policyNet, policyNet

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, inputSize*2+2))        # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def chooseAction(self, GameBoard):
        inputState = convertBoard2PowBoard(torch.from_numpy(np.expand_dims(GameBoard.board, axis=0)))
        # input only one sample
        legalMoves = getLegalMoves(GameBoard)
        if np.random.uniform() <= EPSILON:   # greedy
            actionsQvalue = self.eval_net(inputState)
            # action should first be lgeal actions
            illegalMask = list(set([0, 1, 2, 3]).difference(set(legalMoves)))
            for i in illegalMask: actionsQvalue[0][i] = float('-inf')
            # print("actionsQvalue: ", actionsQvalue)
            action = torch.max(actionsQvalue, 1)[1].data.numpy()[0] # action with maximum Q value
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
        b_s = torch.FloatTensor(b_memory[:, :inputSize])
        b_s = b_s.reshape((BATCH_SIZE, 4, 4)) # reshape back to original 2D board
        b_a = torch.LongTensor(b_memory[:, inputSize:inputSize+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, inputSize+1:inputSize+2])
        b_s_ = torch.FloatTensor(b_memory[:, -inputSize:])
        b_s_ = b_s_.reshape((BATCH_SIZE, 4, 4)) # reshape back to original 2D board

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(convertBoard2PowBoard(b_s)).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(convertBoard2PowBoard(b_s_)).detach()     # detach from graph, don't backpropagate
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

def step(GameBoard, action):
    originalBoard = copy.deepcopy(GameBoard.board)
    originalScore = GameBoard.score
    if action == 0: GameBoard.moveUp()
    elif action == 1: GameBoard.moveDown()
    elif action == 2: GameBoard.moveLeft()
    elif action == 3: GameBoard.moveRight()
    else: assert(False) # should not reach here!

    newBoard = GameBoard.board
    # Current Reward = number of merges + 0.1*log2(new maximum tile number)
    mergeNum = findEmptyTiles(newBoard)-findEmptyTiles(originalBoard)
    GameBoard.addNewTile()
    # GameBoard.printBoard()
    newBoard = GameBoard.board
    maxTile = np.max(newBoard)
    reward = mergeNum+0.1*math.log2(maxTile)
    # print("GameBoard.GameOver()", GameBoard.GameOver())
    if GameBoard.GameOver(): done = True # need further checking
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
        
        # decrease the epsilon value during training
        if (episode > 10000) or (EPSILON > 0.1 and total_iters%2500==0):
            EPSILON = EPSILON/1.005
       
        # store them in memory
        dqn.store_transition(GameBoard, action, reward, newGameBoard)

        episode_reward += reward

        if dqn.memory_counter > MEMORY_CAPACITY:
            print("Learning new experience...")
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


