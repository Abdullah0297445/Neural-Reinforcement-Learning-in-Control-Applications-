#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 02:45:28 2018

@author: fayyaz
"""


import gym
from maglevEnv import MagLevEnv
#import universe
#gym_pull.pull('github.com/Abdullah0297445/simple_mag_lev')        # Only required once, envs will be loaded with import gym_pull afterwards

env = MagLevEnv()
#env = gym.make('simple_mag_lev-v0')

episodes = 10
action_array = [1,1,1,0,1,1,0,0,0,1]


obs = env.reset()

for i in range(episodes):
    obs,reward,done,_ = env.step(action_array[i])
    print(obs,reward,done)
    
    