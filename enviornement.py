import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import random



class MineSweeper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def automaticUncover(self,x,y):
        totalReward = 0
        if self.hiddenChart[y][x] != 9 and self.chart[y][x] < 0: 
            self.chart[y][x] = self.hiddenChart[y][x]
            totalReward += 1
            if self.hiddenChart[y][x] == 0:
                if x + 1 < self.TILE_X_AMOUNT:
                    totalReward += self.automaticUncover(x+1,y)
                if x - 1 >= 0:
                    totalReward += self.automaticUncover(x-1,y)
                if y + 1 < self.TILE_Y_AMOUNT:
                    totalReward += self.automaticUncover(x,y+1)
                if y - 1 >= 0:
                    totalReward += self.automaticUncover(x,y-1)
                if x + 1 < self.TILE_X_AMOUNT and y + 1 < self.TILE_Y_AMOUNT:
                    totalReward += self.automaticUncover(x+1,y+1)
                if x + 1 < self.TILE_X_AMOUNT and y - 1 >= 0:
                    totalReward += self.automaticUncover(x+1,y-1)
                if x - 1 >= 0 and y - 1 >= 0:
                    totalReward += self.automaticUncover(x-1,y-1)
                if x - 1 >= 0 and y + 1 < self.TILE_Y_AMOUNT:
                    totalReward += self.automaticUncover(x-1,y+1)
        return totalReward
    def pickTile(self,x,y):
        if self.firstMove == True:
            #Get all cover tiles outside safety square
            safeTiles = []
            for i in range(0,self.TILE_Y_AMOUNT):
                for j in range (0,self.TILE_X_AMOUNT):
                    if abs(i - y) <= 1 and abs(j - x) <= 1:
                        continue
                    safeTiles.insert(0,(j,i))
            #Place bombs
            placedBombs = 0
            while placedBombs != self.BOMB_AMOUNT:
                    k = random.randint(0,len(safeTiles)-1)
                    self.hiddenChart[safeTiles[k][1]][safeTiles[k][0]] = 9
                    del safeTiles[k] 
                    placedBombs += 1
                    
            #Place numbers near bombs
            for i in range(0,self.TILE_Y_AMOUNT):
                for j in range (0,self.TILE_X_AMOUNT):
                    if self.hiddenChart[i][j] != 9:
                        continue
                    if i - 1 >= 0 and j - 1 >= 0 and self.hiddenChart[i-1][j-1] != 9:
                        self.hiddenChart[i-1][j-1] += 1
                    if i - 1 >= 0 and self.hiddenChart[i-1][j] !=9:
                        self.hiddenChart[i-1][j] += 1
                    if i - 1 >= 0 and j + 1 < self.TILE_X_AMOUNT and self.hiddenChart[i-1][j+1] != 9:
                        self.hiddenChart[i-1][j+1] += 1
                    if j - 1 >= 0 and self.hiddenChart[i][j-1] != 9:
                        self.hiddenChart[i][j-1] += 1
                    if j + 1 < self.TILE_X_AMOUNT and self.hiddenChart[i][j+1] != 9:
                        self.hiddenChart[i][j+1] += 1
                    if i + 1 < self.TILE_Y_AMOUNT and j - 1 >= 0 and self.hiddenChart[i+1][j-1] != 9:
                        self.hiddenChart[i+1][j-1] += 1
                    if i + 1 < self.TILE_Y_AMOUNT and self.hiddenChart[i+1][j] !=9:
                        self.hiddenChart[i+1][j] += 1
                    if i + 1 < self.TILE_Y_AMOUNT and j + 1 < self.TILE_X_AMOUNT and self.hiddenChart[i+1][j+1] != 9:
                        self.hiddenChart[i+1][j+1] += 1
        if self.hiddenChart[y][x] == 9:
            return -1
        totalReward = 0
        #automaticly uncover tiles that are not next to bombs        
        if self.hiddenChart[y][x] == 0:
           totalReward += self.automaticUncover(x,y)
        else:
            totalReward += 1
        self.firstMove = False
        return totalReward
    def __init__(self, renderMode=None, sizeX=20,sizeY=20,bombs=80):
        self.RENDER_MODE = renderMode
        self.TILE_X_AMOUNT = sizeX 
        self.TILE_Y_AMOUNT = sizeY
        self.BOMB_AMOUNT = bombs
        self.WINNING_SCORE = self.TILE_X_AMOUNT * self.TILE_Y_AMOUNT
        #SCREEN_WIDTH = TILE_X_AMOUNT * 32
        #SCREEN_HEIGHT = TILE_Y_AMOUNT * 32

        #Using game screen: Watch game screen
        #self.observation_space = spaces.Box(low=0,high=255,shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3),dtype=np.uint8)
        #Using game logic: Watch grid 0..8 = safe tiles, 9 = bomb, -1 = covered tile
        self.observation_space = spaces.Box(low=-1,high=9,shape=(self.TILE_Y_AMOUNT, self.TILE_X_AMOUNT),dtype=np.int8)
        # Choose any tile on 20x20 grid
        #Action space will shrink overtime, policy must choose only viable tiles/actions (Policy cannot uncover uncovered tile)
        self.action_space = spaces.MultiDiscrete(np.array([self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT]))

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
        self.score = 0
        self.firstMove = True
        
        return np.array([self.chart]).astype(np.int8)
#sample returns y,x picktile needs x y
    def step(self, action):
        reward = self.pickTile(action[1],action[0])
        terminated = False
        if reward == -1:
            reward = 0
            terminated = True
        else:
            self.score += reward
        if self.score == self.WINNING_SCORE:
            terminated = True
        return self.chart, reward, terminated

 

    def render(self,renderMode="human"):
        match renderMode:
            case "human":
                a = 2
            case "console":
                for i in range(0,self.TILE_Y_AMOUNT):
                    for j in range (0,self.TILE_X_AMOUNT):
                        if self.chart[i][j] < 0:
                            print("?",end="")
                        else:
                            print(int(self.chart[i][j]),end="")              
                    print("\n",end="")
    def close(self):
        a = 2