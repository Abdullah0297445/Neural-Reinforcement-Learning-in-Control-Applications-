# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:22:11 2018
A simple example of a magnetic levitation system controlled using Reinforcement Learning
@author: Fayyaz Minhas
"""

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
EPISODES = 80  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01 # e-greedy threshold end value
EPS_DECAY = 1000  # e-greedy threshold decay
GAMMA = 0.98  # Q-learning discount factor
LR = 0.005  # NN optimizer learning rate
BATCH_SIZE = 64  # Q-learning batch size

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
        self.l1 = nn.Linear(4, 200)
        self.l3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = (self.l3(x))
        return x

env = MagLevEnv()
#env.referencepoint = 8.0

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state, actions = None):
    global steps_done
    q_values = []
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        for action in actions:
            q_value = model(Variable(FloatTensor([np.append(state,[action],axis=0)]), volatile=True).type(FloatTensor)).data.view(1, 1)
            q_value = q_value.data.numpy()[0,0]
            q_values.append(q_value)
        return LongTensor([[q_values.index(max(q_values))]])
            
    else:
        return LongTensor([[random.randrange(0,len(actions))]])


def run_episode(e, environment):
    
    action_space = np.linspace(0,1,2)
    state = environment.reset() 
    ref = environment.referencepoint
    state_ref = np.append(state,[ref],axis=0)
    steps = 0
    while True:
        steps += 1
        
        action = select_action(state_ref, action_space)
        state_ref_a = np.append(state_ref,[action.data.numpy()[0,0]],axis=0)
        a = action.data.numpy()[0,0]
        next_state, reward, done, _ = environment.step(a)
        #ref = environment.referencepoint
        next_state_ref = np.append(next_state,[ref],axis=0)
        a_next = select_action(next_state_ref,action_space)
        next_state_ref_a = np.append(next_state_ref,[a_next.data.numpy()[0,0]],axis=0)

        memory.push((FloatTensor([state_ref_a]),
                     action,  # action is already a tensor
                     FloatTensor([next_state_ref_a]),
                     FloatTensor([reward])))

        
        learn()

        state_ref = next_state_ref

        if steps > 500:
            print("Episode %s Final Position Error %s " %(e, np.abs(next_state_ref[1]-ref)))
            episode_durations.append(np.abs(next_state_ref[1]-ref))
            plotError()
            
            break


def learn():
    global GAMMA
    if len(memory) < BATCH_SIZE:
        return
    
    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, LongTensor([[0]]*BATCH_SIZE))
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
        

    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values.view(BATCH_SIZE,1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def plotError():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title("Total Steps Done : " + str(steps_done))
    plt.xlabel('Episode')
    plt.ylabel('Final Position Error')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
#    H = 10
#    if len(durations_t) >= H:
#        means = durations_t.unfold(0, H, 1).mean(1).view(-1)
#        means = torch.cat((torch.zeros(H-1), means))
#        plt.plot(means.numpy())

    plt.pause(0.01)  # pause a bit so that plots are updated



for e in range(EPISODES):
    run_episode(e, env)
    
print('Complete')
#env.render(close=True)
#
#plt.pause(0.5)

#%% TESTING 

action_space = np.linspace(0,1,2)
state = env.reset()

env.position = 0.2
env.velocity = -2.0
#env.mass = 1.0
env.referencepoint = 0.15
state = env._get_state()
state = np.append(state,[env.referencepoint],axis=0)
action = select_action(state,action_space)#.data.numpy()[0,0]
#state = np.append(state,[action.data.numpy()[0,0]],axis=0)
S = [np.append(state,[action.data.numpy()[0,0]],axis=0)] #States history for test

for i in range(500):    
    
    
    state,reward,done,_ = env.step(action.data.numpy()[0,0])
    env.render()
    state = np.append(state,[env.referencepoint],axis=0)
    action = select_action(state,action_space)#.data.numpy()[0,0]
    
    #state = np.append(state,[action.data.numpy()[0,0]],axis=0)
    S.append(np.append(state,[action.data.numpy()[0,0]],axis=0))
    print(i,state,reward,done)
    if done:
        print("out of bounds")
    
S = np.array(S)   

#%% Plotting the policy in the state space.
x = np.linspace(0.0, 0.2032, 50)
v = np.linspace(-10.0, 10.0, 50)
action_space = np.linspace(0,1,2)

A = np.zeros((len(x),len(v)))
for i,xi in enumerate(x):
    for j,vj in enumerate(v):
        A[i,j] = select_action([vj,xi,env.referencepoint], action_space).data.numpy()[0,0]
plt.figure(3)
plt.contourf(v,x,A,levels=[0.1,1]);plt.scatter(S[:,0],S[:,1],c='r'); plt.plot(S[0,0],S[0,1],c = 'k', marker ='*'); plt.scatter(S[-1,0],S[-1,1],c = 'k', marker ='s')
plt.figure(4)
plt.plot(S[:,1])