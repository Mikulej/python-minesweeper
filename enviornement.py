import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces



class MineSweeper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, sizeX=20,sizeY=20):
        self.TILE_X_AMOUNT = sizeX 
        self.TILE_Y_AMOUNT = sizeY
        #SCREEN_WIDTH = TILE_X_AMOUNT * 32
        #SCREEN_HEIGHT = TILE_Y_AMOUNT * 32

        #Using game screen: Watch game screen
        #self.observation_space = spaces.Box(low=0,high=255,shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3),dtype=np.uint8)
        #Using game logic: Watch grid 0..8 = safe tiles, 9 = bomb, -1 = covered tile
        self.observation_space = spaces.Box(low=-1,high=9,shape=(self.TILE_X_AMOUNT, self.TILE_Y_AMOUNT),dtype=np.int8)#SHOULD IT BE REVERSED - X and Y?
        # Choose any tile on 20x20 grid
        #Action space will shrink overtime, policy must choose only viable tiles/actions (Policy cannot uncover uncovered tile)
        self.action_space = spaces.Discrete(self.TILE_X_AMOUNT*self.TILE_Y_AMOUNT)

        self.chart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.hiddenChart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.grid = np.empty((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT),dtype=pygame.Rect)


    def reset(self, seed=None, options=None):
        #initialize agent and reset enviornment
        self.chart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.hiddenChart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.grid = np.empty((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT),dtype=pygame.Rect)

        #create covered tiles
        for i in range(0,self.TILE_Y_AMOUNT):
            for j in range (0,self.TILE_X_AMOUNT):
                self.chart[i][j] = -1
                self.grid[i][j] = pygame.Rect(j * 32,i*32,32,32).copy()
        
        return observation, info

    def step(self, action):
        a = 2
        return observation, reward, terminated, truncated, info

 

    def render(self):
        a = 2

    def close(self):
        a = 2