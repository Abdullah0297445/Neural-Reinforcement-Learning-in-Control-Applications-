import gym
from gym import error, spaces, utils
from gym.utils import seeding
import logging
import numpy as np


class MagLevEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    GRAVITY = 9.8
    FORCE = GRAVITY*2
    
    
    
    def __init__(self, mass = 1, referencepoint = 7, timestep=0.01):
        
        self.__version__ = "0.0.1.0"
        logging.info("MAGLevEnv - Version {}".format(self.__version__))
        
        self.timestep = timestep
        self.mass = mass
        
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([-9.9,-100,0]), np.array([9.9,100, 10]), dtype=np.float32)
        
        self.action_episode_memory = []
        
        self.acceleration = 0
        self.velocity = initialvel 
        self.position = initialpos
        
        self.initialpos = initialpos
        self.initialvel = initialvel
        
        #self.observation = []
        self.rewardtype = rewardtype

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
        self._take_action(action)
        done = False
        reward = self._get_reward()
        obs = self._get_state()
        
        if not self.observation_space.contains(obs):
            done = True
        
        return obs, reward, done, {}
        
        


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        
        self.action_episode_memory = []
        
        self.acceleration = 0
        self.velocity = self.initialvel
        self.position = self.initialpos
       
        return np.asarray([self.acceleration,self.velocity,self.position])

    def render(self):
        
        plt.figure(0)
        circle = plt.Circle((0,self.position), radius= 0.6, color = 'r')
        ax=plt.gca()
        ax.clear()
        ax.add_patch(circle)
        plt.axis('scaled')
        plt.xlim(-15,15)
        plt.ylim(-15,15)
        
        plt.pause(0.00001)
        plt.show()

    def _take_action(self, action):
        
        v0 = self.velocity
        x0 = self.position
            
        
        
        a = ( ( action*MagLevEnv.FORCE / self.mass ) - MagLevEnv.GRAVITY )
         
        dv = ( a * self.timestep )
        v = v0 + dv
        dx = ( v0 * self.timestep ) + 0.5 * (a * self.timestep**2) 
        x = x0 + dx
    
        
        self.acceleration = a
        self.velocity = v
        self.position = x
        #self.observation.append(np.asarray(list((self.acceleration,self.velocity,self.position))))
        
        self.action_episode_memory.append(action)
        
            
    def _get_state(self):
        
        """Get the observation."""
        
        obs = np.asarray(list((self.acceleration,self.velocity,self.position)))
        
        
        return obs
            
            
                             
        

    def _get_reward(self):
        if float(self.position) >= 4.0 and float(self.position) <= 10.0:
            if self.rewardtype == 'parabolic':
                reward = (1-(1/9)*abs(self.position-self.referencepoint)**2)
            else:
                reward = (1-(1/3)*abs(self.position-self.referencepoint))
            
        else:
            reward = -1
            
        return reward
