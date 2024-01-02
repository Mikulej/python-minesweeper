import pygame
import random
import numpy as np
from sys import exit


pygame.init()
screen = pygame.display.set_mode((768,640))
TILE_X_AMOUNT = int(768 / 32)
TILE_Y_AMOUNT = int(640 / 32)
BOMB_AMOUNT = 20
pygame.display.set_caption('Minesweeper')
clock = pygame.time.Clock()
#choose difficulty mode easy/medium/hard harder=the more tiles and bombs


sprites = pygame.image.load("sprites/minesweeper.png")
sprites = pygame.transform.scale(sprites,(128,128))
#use rectangle for collision detection(mouse point)

def getSprite(spriteSheet,w,h,x,y):
    sprite = pygame.Surface((w,h)).convert_alpha()
    sprite.blit(spriteSheet,(0,0),(w * x, h * y,w,h))
    return sprite

def drawTile(x,y,id):
    match id:
        case -2:
            screen.blit(spriteEmpty2,(x*32,y*32))
        case -1: 
            screen.blit(spriteEmpty1,(x*32,y*32))
        case 0:
            screen.blit(sprite0,(x*32,y*32))
        case 9:
            screen.blit(spriteBomb,(x*32,y*32))
        

class Tile:
    discovered = -2
    id = 0
    def __init__(self,id) -> None:
        self.id = id


    

spriteEmpty1 = getSprite(sprites,32,32,0,0)
spriteEmpty2 = getSprite(sprites,32,32,1,0)
sprite0 = getSprite(sprites,32,32,3,3)
spriteBomb = getSprite(sprites,32,32,0,3)

chart = np.zeros((TILE_X_AMOUNT,TILE_Y_AMOUNT))
hiddenChart = np.zeros((TILE_X_AMOUNT,TILE_Y_AMOUNT))
grid = np.empty((TILE_X_AMOUNT,TILE_Y_AMOUNT),dtype=pygame.Rect)

#Create cover tiles
for i in range(0,TILE_X_AMOUNT):
    #chart[i] = random.randint(0,1)
    for j in range (0,TILE_Y_AMOUNT):
        #chart[i][j] = random.randint(0,1)
        if i%2==0:
            if j%2==0:
                chart[i][j] = -1
            else:
                chart[i][j] = -2
        else:
            if j%2==0:
                chart[i][j] = -2
            else:
                chart[i][j] = -1
            print(chart[i][j])
        
        grid[i][j] = pygame.Rect(i * 32,j*32,32,32).copy()
        print(i, " ", j,"|")
        # if chart[i][j] == 0:
        #     print("O",end="")
        # else:
        #     print("X",end="")
    print("\n",end="")

#Create content of tiles
#Place BOMB_AMOUNT bombs
placedBombs = 0
while placedBombs != BOMB_AMOUNT:
    while True:
        x = random.randint(0,TILE_X_AMOUNT - 1)
        y = random.randint(0,TILE_Y_AMOUNT - 1)
        if hiddenChart[x][y] != 9:
            hiddenChart[x][y] = 9
            placedBombs += 1
            break

#screen.blit(spriteBomb,(100,0))
for bomb in hiddenChart:
    print(bomb)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx,my = pygame.mouse.get_pos()
            chart[int(mx/32)][int(my/32)] = hiddenChart[int(mx/32)][int(my/32)]


    screen.fill((0,0,0))

    for i in range(0,TILE_X_AMOUNT):
        for j in range (0,TILE_Y_AMOUNT):
            drawTile(i,j,chart[i][j])         
    

 
    #render loop
    pygame.display.update()
    clock.tick(60) #60 framerate cap


#Zrodla
#https://www.youtube.com/watch?v=M6e3_8LHc7A&list=WL&index=84&t=662s getSprite
#https://www.youtube.com/watch?v=AY9MnQ4x3zk&t=8207s main