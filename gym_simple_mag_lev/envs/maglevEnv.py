import gym
from gym import spaces, utils, error
from gym.utils import seeding
import logging
import numpy as np
import matplotlib.pyplot as plt
import random

class MagLevEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    
    GRAVITY = 9.8
    FORCE = GRAVITY*0.75
    
    
    
    def __init__(self):
        
        self.__version__ = "0.0.1.0"
        logging.info("MAGLevEnv - Version {}".format(self.__version__))
        
        self.timestep = 0.01
        self.mass = 0.35615
        
        #Observation and Action spaces.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([-6.0,0.0]), np.array([6.0,0.2032]))
    
        
        self.acceleration = 0
        self.velocity = 0 
        
        self.estimatedvel = 0
        self.position = 0
        self.referencepoint = 0.1
        self.lastAction = 0
        #self.I_sq = 0
        self.magpos = 0.2032
        #self.k = 13440
        #
        
        
    def step(self, action):
        """

        Parameters
        ----------
        action : any action. In our case '1' or '0' as int value.

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
        self.lastAction = action
        done = False
        reward = self._get_reward()
        obs = self._get_state()
        
        if not self.observation_space.contains(obs):
            done = True
        
        return obs, reward, done, {}
        
        


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        
        Returns a numpy array ([velocity,position])
        -------
        observation (object): the initial observation of the space.
        """
        
        #Randomly set position of our object(solid metallic ball).
        self.mass = random.uniform(0.05, 0.5)
        x = random.uniform(0.0, 0.2032)
        self.position = x
        
        r = random.uniform(0.05, 0.1950)
        self.referencepoint = r 
        
        #Randomly set velocity of our object(solid metallic ball).
        v = (2*np.random.rand()-1)*6
        self.velocity = v
        
        self.acceleration = 0
        
        #xerror = -np.abs(self.referencepoint - self.position)
        
        return np.asarray([self.velocity,self.position])

    def render(self, figid = 0):
        
        """
        Shows a ball with the position indicated by the position. Velocity is 
        proportional to the size of the ball. The current action is shown in 
        color of the ball with blue indicating no upward force and red if the
        force is active.
        """
        
        plt.figure(figid)
        r = np.max((0.3,np.abs(self.velocity)/6.0))
        c = 'b'
        if self.lastAction:
            c = 'r'
        
        
        circle = plt.Circle((0,self.position*39.37), radius= r, color = c)
        
        ax=plt.gca()
        ax.clear()
        ax.add_patch(circle)
        plt.axis('scaled')
        plt.xlim(-5,5)
        plt.ylim(-1,10)
        plt.plot([-5,5],[self.referencepoint*39.37]*2)
        plt.plot([-5,5],[0]*2)
        plt.plot([-5,5],[8]*2)
        plt.title("Rendering")
        plt.xlabel(' ')
        plt.ylabel('Position (in inches)')
        plt.legend(['Ref Point','Ground','Magnet'], bbox_to_anchor=(1, 1),
          ncol=1)
        plt.pause(0.00001)
        plt.show()

    def _take_action(self, action):
        
        """
        Virtually performs the action using the physics of our problem. It is a helper function used by our Step function.
        
        If choosen action is '1' then Force is applied for 0.01 sec if action is '0' no force is applied in that timestep (0.01 sec).
        
        This method updates the values of current_position and current_velocity in our environment.
        
        Returns Nothing.
        -------
        
        Parameters: Action '1' or '2' as an int value. 
        """
        #self.I = action
        
        v0 = self.velocity
        x0 = self.position 
        k = 13440
        r = self.magpos - x0
        
        I_sq = ( MagLevEnv.FORCE * (r)**2 ) / k
        I_sq = min(2, I_sq)
        Force = k * I_sq / (r)**2 
        #Force = MagLevEnv.FORCE
        
        if (r) < 0:
            Force = -1 * Force
        
        a = ( ( action * Force / self.mass ) - MagLevEnv.GRAVITY )
         
        dv = ( a * self.timestep )
        v = v0 + dv
        dx = ( v0 * self.timestep ) + 0.5 * (a * self.timestep**2) 
        x = x0 + dx
        
        self.estimatedvel = ( float(x) - float(x0) )/ self.timestep
        
        self.acceleration = a
        self.velocity = v
        self.position = x
            
    def _get_state(self):
        
        """
        Get the current state of the environment (observation).
        
        State in our case is just a numpy array of ([current_velocity, current_position])
        
        It is a helper function for our Step method.
        
        Returns a numpy array ([velocity,position])
        -------
        Parameters: Empty.
        """
        #xerror = -np.abs(self.referencepoint - self.position)
        obs = np.asarray([self.velocity,self.position])
        
        
        return obs
            
            
                             
        

    def _get_reward(self):
        
        state = self._get_state()
        reward =  -np.abs(float(state[1] - self.referencepoint)) #*float(np.abs(next_state[0])<0.1)
        if np.abs(float(state[1] - self.referencepoint))<0.01:
            reward+=2.0
            if abs(self.velocity)<0.2:
                reward+=1.0
        if not self.observation_space.contains(state):
            reward-=1.0            
        return reward
