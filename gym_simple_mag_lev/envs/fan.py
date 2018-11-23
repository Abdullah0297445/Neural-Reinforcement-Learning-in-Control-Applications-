import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from maglevEnv import MagLevEnv
import numpy as np

# hyper parameters
EPISODES = 50  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01 # e-greedy threshold end value
EPS_DECAY = 1000  # e-greedy threshold decay
GAMMA = 0.98  # Q-learning discount factor
LR = 0.005  # NN optimizer learning rate
HIDDEN_LAYER = 200  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
ALPHA = 0.005

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)
        
        #self.l3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

env = MagLevEnv()


model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []
abs_from_target = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while steps<300:
        steps += 1
        #environment.render()
        action = select_action(FloatTensor([state]))
        a = action.data.numpy()[0,0]
        next_state, reward, done, _ = environment.step(a)

        # zero reward when attempt ends
#        if done and steps < 200:
#            reward = 0

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        
        learn()

        state = next_state

        if steps == 299:
            print("Episode %s finished after %s steps" %(e, steps))
            episode_durations.append(steps)
            abs_from_target.append(abs(env.position - env.referencepoint))
            plot_durations()
            


def learn():
    
    if len(memory) < BATCH_SIZE:
        return
    
    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state),requires_grad=False)

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1,batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
        

    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(BATCH_SIZE,1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def plot_durations():
    plt.figure(2)
    plt.clf()
    distance_y = torch.FloatTensor(abs_from_target)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Distance_FROM_Target')
    plt.plot(distance_y.numpy())
    # take 100 episode averages and plot them too
    if len(distance_y) >= 100:
        means = distance_y.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
#env.render(close=True)
#
1/0
state = env.reset()
env.initialpos = 0.0

for i in range(600):
    action = select_action(FloatTensor([state]))
    a = action.data.numpy()[0,0]
    state,reward,done,_ = env.step(a)
    print(i,state,a,reward,done)
    env.render()
