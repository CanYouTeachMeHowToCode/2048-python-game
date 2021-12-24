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

# convert the input game board into corresponding power of 2 tensor.
def convertBoard2PowBoard(board):
    powerBoard = np.zeros(shape=(1, 16, len(board), len(board)), dtype=np.float32) # 16 is for board size 4 (2^0 to 2^15)
    for i in range(len(board)):
        for j in range(len(board)):
            if not board[i][j]:
                powerBoard[0][0][i][j] = 1.0
            else:
                power = int(math.log(board[i][j],2))
                powerBoard[0][power][i][j] = 1.0
    return powerBoard # NCHW format, different from NHWC in tf

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

# unit test
GameBoard = Board(4)
print(GameBoard.board)
print(convertBoard2PowBoard(GameBoard.board))
GameBoard.moveLeft()
print(GameBoard.board)
print(convertBoard2PowBoard(GameBoard.board))
GameBoard.addNewTile()
print(GameBoard.board)
print(convertBoard2PowBoard(GameBoard.board))

print("GameBoard.board", GameBoard.board)
testInput = torch.from_numpy(convertBoard2PowBoard(GameBoard.board))
res = policyNet(testInput)
print("res", res)
print("torch.max(res, 1)[1].data.numpy()", torch.max(res, 1)[1].data.numpy())

# Hyper Parameters
LR = 5e-4 # learning rate
GAMMA = 0.9 # reward discount
EPSILON = 0.9 # for epsilon greedy algorithm
REPLAY_MEMORY, REPLAY_LABELS = [], []

MEMORY_CAPACITY = 6000

loss = []
scores = []
final_parameters = {}
num_episodes = 20000


# for episode with max score
maximum = -1
episode = -1

# total iterations 
total_iters = 1

# # number of back props
# back = 0

# tensor = torch.ones(())
# single_dataset = tensor.new_empty((1, 16, 4, 4), dtype=torch.float32) # tf.placeholder(tf.float32,shape=(1,4,4,16))
# print("single_dataset", single_dataset)
# single_output = policy(single_dataset)


## 改好这部分
class DQN:
    def __init__(self):
        self.eval_net, self.target_net = policyNet, policyNet

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def chooseAction(self, GameBoard):
        inputState = torch.from_numpy(convertBoard2PowBoard(GameBoard.board))
        # input only one sample
        if np.random.uniform() <= EPSILON:   # greedy
            actionsQvalue = self.eval_net(inputState)
            action = torch.max(actionsQvalue, 1)[1].data.numpy()[0] # action with maximum Q value
        else:   # random
            action = np.random.randint(0, 4)
        return action

    def performAction(self, GameBoard, action):
        if action == 0: GameBoard.moveUp()
        elif action == 1: GameBoard.moveDown()
        elif action == 2: GameBoard.moveLeft()
        elif action == 3: GameBoard.moveRight()
        else: assert(False) # should not reach here!

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
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
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

#读懂code在干啥再改写
for episode in range(num_episodes):
    # start state
    GameBoard = Board(4)
    GameBoard.addNewTile()
    
    # whether current episode finished or not
    finish = False
    
    # total score of this episode
    total_score = 0
    
    while(not finish):
        # choose action
        action = dqn.chooseAction(s)

        # take action
        # (continue here)

        ## s_, r, done, info = env.step(a)
        
        # decrease the epsilon value during training
        if (episode > 10000) or (EPSILON > 0.1 and total_iters%2500==0):
            EPSILON = EPSILON/1.005
            
       
        #change the matrix values and store them in memory
        prev_state = deepcopy(originalBoard)
        prev_state = change_values(prev_state)
        prev_state = np.array(prev_state,dtype=np.float32).reshape(1,4,4,16)
        replay_labels.append(labels)
        replay_memory.append(prev_state)
        
        
        #back-propagation
        if(len(replay_memory)>=mem_capacity):
            back_loss = 0
            batch_num = 0
            z = list(zip(replay_memory,replay_labels))
            np.random.shuffle(z)
            np.random.shuffle(z)
            replay_memory,replay_labels = zip(*z)
            
            for i in range(0,len(replay_memory),batch_size):
                if(i + batch_size>len(replay_memory)):
                    break
                    
                batch_data = deepcopy(replay_memory[i:i+batch_size])
                batch_labels = deepcopy(replay_labels[i:i+batch_size])
                
                batch_data = np.array(batch_data,dtype=np.float32).reshape(batch_size,4,4,16)
                batch_labels = np.array(batch_labels,dtype=np.float32).reshape(batch_size,output_units)
            
                feed_dict = {tf_batch_dataset: batch_data, tf_batch_labels: batch_labels}
                _,l = session.run([optimizer,loss],feed_dict=feed_dict)
                back_loss += l 
                
                print("Mini-Batch - {} Back-Prop : {}, Loss : {}".format(batch_num,back,l))
                batch_num +=1
            back_loss /= batch_num
            J.append(back_loss)
            
            #store the parameters in a dictionary
            final_parameters['conv1_layer1_weights'] = session.run(conv1_layer1_weights)
            final_parameters['conv1_layer2_weights'] = session.run(conv1_layer2_weights)
            final_parameters['conv2_layer1_weights'] = session.run(conv2_layer1_weights)
            final_parameters['conv2_layer2_weights'] = session.run(conv2_layer2_weights)
            final_parameters['conv1_layer1_biases'] = session.run(conv1_layer1_biases)
            final_parameters['conv1_layer2_biases'] = session.run(conv1_layer2_biases)
            final_parameters['conv2_layer1_biases'] = session.run(conv2_layer1_biases)
            final_parameters['conv2_layer2_biases'] = session.run(conv2_layer2_biases)
            final_parameters['fc_layer1_weights'] = session.run(fc_layer1_weights)
            final_parameters['fc_layer2_weights'] = session.run(fc_layer2_weights)
            final_parameters['fc_layer1_biases'] = session.run(fc_layer1_biases)
            final_parameters['fc_layer2_biases'] = session.run(fc_layer2_biases)
            
            #number of back-props
            back+=1
            
            #make new memory 
            replay_memory = list()
            replay_labels = list()
            
        
        if(local_iters%400==0):
            print("Episode : {}, Score : {}, Iters : {}, Finish : {}".format(ep,total_score,local_iters,finish))
        
        local_iters += 1
        total_iters += 1
        
    scores.append(total_score)
    print("Episode {} finished with score {}, result : {} board : {}, epsilon  : {}, learning rate : {} ".format(ep,total_score,finish,board,epsilon,session.run(learning_rate)))
    print()
    
    if((ep+1)%1000==0):
        print("Maximum Score : {} ,Episode : {}".format(maximum,episode))    
        print("Loss : {}".format(J[len(J)-1]))
        print()
        
    if(maximum<total_score):
        maximum = total_score
        episode = ep
print("Maximum Score : {} ,Episode : {}".format(maximum,episode))    



