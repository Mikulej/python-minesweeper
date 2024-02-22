import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import random

from typing import List

class MineSweeper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    sprites = pygame.image.load("sprites/minesweeper.png")
    sprites = pygame.transform.scale(sprites,(128,128))

    def getSprite(self,spriteSheet,w,h,x,y):
        sprite = pygame.Surface((w,h)).convert_alpha()
        sprite.blit(spriteSheet,(0,0),(w * x, h * y,w,h))
        return sprite
   

    def drawTile(self,x,y,id):
        match id:
            case -2:
                self.screen.blit(self.spriteEmpty2,(x*32,y*32))
            case -1: 
                self.screen.blit(self.spriteEmpty1,(x*32,y*32))
            case 0:
                self.screen.blit(self.sprite0,(x*32,y*32))
            case 1:
                self.screen.blit(self.sprite1,(x*32,y*32))
            case 2:
                self.screen.blit(self.sprite2,(x*32,y*32))
            case 3:
                self.screen.blit(self.sprite3,(x*32,y*32))
            case 4:
                self.screen.blit(self.sprite4,(x*32,y*32))
            case 5:
                self.screen.blit(self.sprite5,(x*32,y*32))
            case 6:
                self.screen.blit(self.sprite6,(x*32,y*32))
            case 7:
                self.screen.blit(self.sprite7,(x*32,y*32))
            case 8:
                self.screen.blit(self.sprite8,(x*32,y*32))
            case 9:
                self.screen.blit(self.spriteBomb,(x*32,y*32))    
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
            self.chart[y][x] = self.hiddenChart[y][x]
            totalReward += 1
        self.firstMove = False
        return totalReward
    def __init__(self, renderMode=None, sizeX=20,sizeY=20,bombs=80):
        self.RENDER_MODE = renderMode
        self.TILE_X_AMOUNT = sizeX 
        self.TILE_Y_AMOUNT = sizeY
        self.BOMB_AMOUNT = bombs
        self.WINNING_SCORE = (self.TILE_X_AMOUNT * self.TILE_Y_AMOUNT) - self.BOMB_AMOUNT
        #SCREEN_WIDTH = TILE_X_AMOUNT * 32
        #SCREEN_HEIGHT = TILE_Y_AMOUNT * 32

        #Using game screen: Watch game screen
        #self.observation_space = spaces.Box(low=0,high=255,shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3),dtype=np.uint8)
        #Using game logic: Watch grid 0..8 = safe tiles, 9 = bomb, -1 = covered tile
        self.observation_space = spaces.Box(low=-1,high=9,shape=(self.TILE_Y_AMOUNT, self.TILE_X_AMOUNT),dtype=np.int8)
        # Choose any tile on 20x20 grid
        #Action space will shrink overtime, policy must choose only viable tiles/actions (Policy cannot uncover uncovered tile)
        self.action_space = spaces.Discrete(self.TILE_Y_AMOUNT*self.TILE_X_AMOUNT)
        
        self.possible_actions = []
        for i in range(0,self.TILE_Y_AMOUNT):
            for j in range (0,self.TILE_X_AMOUNT):
                self.possible_actions.append((i*self.TILE_X_AMOUNT)+j)
        self.invalid_actions = []

        self.chart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.hiddenChart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.grid = np.empty((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT),dtype=pygame.Rect)
        self.score = 0
        if renderMode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.TILE_X_AMOUNT * 32,self.TILE_Y_AMOUNT * 32))
            pygame.display.set_caption('Minesweeper')
            self.sprites = pygame.image.load("sprites/minesweeper.png")
            self.sprites = pygame.transform.scale(self.sprites,(128,128))
            self.spriteEmpty1 = self.getSprite(self.sprites,32,32,0,0)
            self.spriteEmpty2 = self.getSprite(self.sprites,32,32,1,0)
            self.sprite0 = self.getSprite(self.sprites,32,32,3,3)
            self.sprite1 = self.getSprite(self.sprites,32,32,0,1)
            self.sprite2 = self.getSprite(self.sprites,32,32,1,1)
            self.sprite3 = self.getSprite(self.sprites,32,32,2,1)
            self.sprite4 = self.getSprite(self.sprites,32,32,3,1)
            self.sprite5 = self.getSprite(self.sprites,32,32,0,2)
            self.sprite6 = self.getSprite(self.sprites,32,32,1,2)
            self.sprite7 = self.getSprite(self.sprites,32,32,2,2)
            self.sprite8 = self.getSprite(self.sprites,32,32,3,2)
            self.spriteBomb = self.getSprite(self.sprites,32,32,0,3)


    def reset(self, seed=None, options=None):
        #return info message
        info = {
            "state": "Lost",
            "score": str(self.score) + "/" + str(self.WINNING_SCORE)
        }
        if self.score == self.WINNING_SCORE:
            info["state"] = "Won"
        #info = "Terminated with score: " + str(self.score) + "/" + str(self.WINNING_SCORE)
        #initialize agent and reset enviornment
        self.chart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.hiddenChart = np.zeros((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT))
        self.grid = np.empty((self.TILE_Y_AMOUNT,self.TILE_X_AMOUNT),dtype=pygame.Rect)
        self.invalid_actions = []

        #create covered tiles
        for i in range(0,self.TILE_Y_AMOUNT):
            for j in range (0,self.TILE_X_AMOUNT):
                self.chart[i][j] = -1
                self.grid[i][j] = pygame.Rect(j * 32,i*32,32,32).copy()
        self.score = 0
        self.firstMove = True
        return np.array(self.chart).astype(np.int8), info

    def decode_action_y(self,action):
        y = int(action / self.TILE_Y_AMOUNT)
        return y
    def decode_action_x(self,action):
        x = action % self.TILE_X_AMOUNT
        return x
    #sample returns y,x picktile needs x y
    def step(self, action):
        #return info message
        info = {
            "state": "Playing",
            "score": str(self.score) + "/" + str(self.WINNING_SCORE)
        }
        reward = self.pickTile(self.decode_action_x(action),self.decode_action_y(action))
        self.update_invalid_actions()
        terminated = False
        truncated = False
        if reward == -1:
            self.revealChart()
            reward = 0
            terminated = True
        else:
            self.score += reward
        if self.score == self.WINNING_SCORE:
            truncated = True
        return np.int8(self.chart), reward, terminated, truncated, info

    def render(self,renderMode="human"):
        match renderMode:
            case "human":
                self.screen.fill((0,0,0))
                for i in range(0,self.TILE_Y_AMOUNT):
                    for j in range (0,self.TILE_X_AMOUNT):
                        self.drawTile(j,i,self.chart[i][j])  
                
                #self.screen.blit(,(lastX*10,lastY*10)
                pygame.display.update()
                #clock.tick(60) #60 framerate cap      
            case "console":
                for i in range(0,self.TILE_Y_AMOUNT):
                    for j in range (0,self.TILE_X_AMOUNT):
                        if self.chart[i][j] < 0:
                            print("?",end="")
                        else:
                            print(int(self.chart[i][j]),end="")              
                    print("\n",end="")
    def close(self):
        pygame.quit()
        exit()
    def update_invalid_actions(self):
        self.invalid_actions = []
        for i in range(0,self.TILE_Y_AMOUNT):
            for j in range (0,self.TILE_X_AMOUNT):
                if self.chart[i][j] >= 0:
                    self.invalid_actions.append((i*self.TILE_X_AMOUNT)+j)
        #print("Invalid actions from update_invalid_actions:")
        #print(self.invalid_actions)

    def action_masks(self) -> List[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]
    
    def revealChart(self):
        for i in range(0,self.TILE_Y_AMOUNT):
            for j in range (0,self.TILE_X_AMOUNT):
                self.chart[i][j] = self.hiddenChart[i][j]
    

    
       
        

    
        
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.rst