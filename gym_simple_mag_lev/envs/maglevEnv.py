import gym
from gym import error, spaces, utils
from gym.utils import seeding
import logging
import numpy as np


class MagLevEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    GRAVITY = 9.8
    FORCE = 10
    
    
    
    def __init__(self, mass = 1, referencepoint = 7, timestep=0.01):
        
        self.__version__ = "0.0.1.0"
        logging.info("MAGLevEnv - Version {}".format(self.__version__))
        self.timestep = timestep
        self.mass = mass
        self.curr_episode = -1
        self.curr_step = -1
        
        self.action_space = spaces.Discrete(2)
        
        
        self.observation_space = spaces.Box(np.array([0, -100]), np.array([10, 100]), dtype=np.float32)
        
        self.action_episode_memory = []
        self.AVP_memory = []
        
               
        #self.reference_once_achieved = False
        #self.current_position = 0.00
        self.referencepoint = referencepoint
        
    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        obs, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.curr_step += 1
        self._take_action(action)
        done = False
        reward = self._get_reward()
        obs = self._get_state()
        if not self.observation_space.contains(obs[1:]):
            done = True
            #self.curr_episode += 1
            self.curr_step = -1
        
        return obs, reward, done, {}
        
        


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode = 0
        self.action_episode_memory = [[]]
        self.AVP_memory = [[]]
        
        
        return np.asarray([0.0,0.0,0.0])

    def render(self):
        pass

    def _take_action(self, action):
        if action == 1:
            if self.curr_step == 0:
                a0 = 0.0
                v0 = 0.0
                x0 = 0.0
            else:
                a0 = self.AVP_memory[self.curr_episode][self.curr_step-1][0]
                v0 = self.AVP_memory[self.curr_episode][self.curr_step-1][1]
                x0 = self.AVP_memory[self.curr_episode][self.curr_step-1][2]
                
            
            a = ( ( MagLevEnv.FORCE / self.mass ) - MagLevEnv.GRAVITY )
            dv = ( a * self.timestep )
            dx = ( dv * self.timestep ) + 0.5 * (MagLevEnv.GRAVITY * self.timestep**2) 
            
            #self.current_position += dx
            
            self.AVP_memory[self.curr_episode].append((a0 + a, v0 + dv, x0 + dx))
            self.action_episode_memory[self.curr_episode].append(1)
        else:
            if self.curr_step == 0:
                a0 = 0.0
                v0 = 0.0
                x0 = 0.0
            else:
                a0 = self.AVP_memory[self.curr_episode][self.curr_step-1][0]
                v0 = self.AVP_memory[self.curr_episode][self.curr_step-1][1]
                x0 = self.AVP_memory[self.curr_episode][self.curr_step-1][2]
            
            a = ( ( 0.0 / self.mass ) - MagLevEnv.GRAVITY )
            dv = ( a * self.timestep )
            dx = ( dv * self.timestep ) + 0.5 * (MagLevEnv.GRAVITY * self.timestep**2)
            
            self.AVP_memory[self.curr_episode].append((a0 + a , v0 + dv, x0 + dx))
            self.action_episode_memory[self.curr_episode].append(0)
            
    def _get_state(self):
        
        """Get the observation."""
        
        ob = np.asarray(list(self.AVP_memory[self.curr_episode][self.curr_step]))
        
        
        return ob
            
            
                             
        

    def _get_reward(self):
        pass
