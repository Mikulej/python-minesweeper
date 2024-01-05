import pygame
import random
import numpy as np
from sys import exit

pygame.init()
TILE_X_AMOUNT = 20
TILE_Y_AMOUNT = 20
screen = pygame.display.set_mode((TILE_X_AMOUNT * 32,TILE_Y_AMOUNT * 32))

BOMB_AMOUNT = 50
SAFE_TILES = TILE_X_AMOUNT * TILE_Y_AMOUNT - BOMB_AMOUNT
pygame.display.set_caption('Minesweeper')
clock = pygame.time.Clock()
firstMove = True
#choose difficulty mode easy/medium/hard harder=the more tiles and bombs
chart = np.zeros((TILE_Y_AMOUNT,TILE_X_AMOUNT))
hiddenChart = np.zeros((TILE_Y_AMOUNT,TILE_X_AMOUNT))
grid = np.empty((TILE_Y_AMOUNT,TILE_X_AMOUNT),dtype=pygame.Rect)
score = 0

sprites = pygame.image.load("sprites/minesweeper.png")
sprites = pygame.transform.scale(sprites,(128,128))

def getSprite(spriteSheet,w,h,x,y):
    sprite = pygame.Surface((w,h)).convert_alpha()
    sprite.blit(spriteSheet,(0,0),(w * x, h * y,w,h))
    return sprite

spriteEmpty1 = getSprite(sprites,32,32,0,0)
spriteEmpty2 = getSprite(sprites,32,32,1,0)
sprite0 = getSprite(sprites,32,32,3,3)
sprite1 = getSprite(sprites,32,32,0,1)
sprite2 = getSprite(sprites,32,32,1,1)
sprite3 = getSprite(sprites,32,32,2,1)
sprite4 = getSprite(sprites,32,32,3,1)
sprite5 = getSprite(sprites,32,32,0,2)
sprite6 = getSprite(sprites,32,32,1,2)
sprite7 = getSprite(sprites,32,32,2,2)
sprite8 = getSprite(sprites,32,32,3,2)
spriteBomb = getSprite(sprites,32,32,0,3)

def drawTile(x,y,id):
    match id:
        case -2:
            screen.blit(spriteEmpty2,(x*32,y*32))
        case -1: 
            screen.blit(spriteEmpty1,(x*32,y*32))
        case 0:
            screen.blit(sprite0,(x*32,y*32))
        case 1:
            screen.blit(sprite1,(x*32,y*32))
        case 2:
            screen.blit(sprite2,(x*32,y*32))
        case 3:
            screen.blit(sprite3,(x*32,y*32))
        case 4:
            screen.blit(sprite4,(x*32,y*32))
        case 5:
            screen.blit(sprite5,(x*32,y*32))
        case 6:
            screen.blit(sprite6,(x*32,y*32))
        case 7:
            screen.blit(sprite7,(x*32,y*32))
        case 8:
            screen.blit(sprite8,(x*32,y*32))
        case 9:
            screen.blit(spriteBomb,(x*32,y*32))    

def peek():
    for i in range(0,TILE_Y_AMOUNT):
        for j in range (0,TILE_X_AMOUNT):
            if hiddenChart[i][j] == 9:
                print("X",end="")
            else:
                print(int(hiddenChart[i][j]),end="")
        print("")

def revealBombs():
    for i in range(0,TILE_Y_AMOUNT):
        for j in range (0,TILE_X_AMOUNT):
            if hiddenChart[i][j] == 9:
                chart[i][j] = hiddenChart[i][j]

def bombsAround(x,y):
    bombAmount = 0
    if y - 1 >= 0 and x - 1 >= 0 and hiddenChart[y-1][x-1] == 9:
        bombAmount += 1
    if y - 1 >= 0 and hiddenChart[y-1][x] ==9:
         bombAmount += 1
    if y - 1 >= 0 and x + 1 < TILE_X_AMOUNT and hiddenChart[y-1][x+1] == 9:
         bombAmount += 1
    if x - 1 >= 0 and hiddenChart[y][x-1] == 9:
         bombAmount += 1
    if x + 1 < TILE_X_AMOUNT and hiddenChart[y][x+1] == 9:
         bombAmount += 1
    if y + 1 < TILE_Y_AMOUNT and x - 1 >= 0 and hiddenChart[y+1][x-1] == 9:
         bombAmount += 1
    if y + 1 < TILE_Y_AMOUNT and hiddenChart[y+1][x] ==9:
         bombAmount += 1
    if y + 1 < TILE_Y_AMOUNT and x + 1 < TILE_X_AMOUNT and hiddenChart[y+1][x+1] == 9:
         bombAmount += 1
    return bombAmount

def automaticUncover(x,y):
    global score
    if hiddenChart[y][x] != 9 and chart[y][x] < 0: 
        chart[y][x] = hiddenChart[y][x]
        score += 1
        if hiddenChart[y][x] == 0:
            if x + 1 < TILE_X_AMOUNT:
                automaticUncover(x+1,y)
            if x - 1 >= 0:
                automaticUncover(x-1,y)
            if y + 1 < TILE_Y_AMOUNT:
                automaticUncover(x,y+1)
            if y - 1 >= 0:
                automaticUncover(x,y-1)
            if x + 1 < TILE_X_AMOUNT and y + 1 < TILE_Y_AMOUNT:
                automaticUncover(x+1,y+1)
            if x + 1 < TILE_X_AMOUNT and y - 1 >= 0:
                automaticUncover(x+1,y-1)
            if x - 1 >= 0 and y - 1 >= 0:
                automaticUncover(x-1,y-1)
            if x - 1 >= 0 and y + 1 < TILE_Y_AMOUNT:
                automaticUncover(x-1,y+1)
    
def pickTile(x,y):
    global firstMove
    global score
    if firstMove == True:
        replacedBombs = 0
        if y - 1 >= 0 and x - 1 >= 0 and hiddenChart[y-1][x-1] == 9:
            hiddenChart[y-1][x-1] = 0
            replacedBombs += 1
        if y - 1 >= 0 and hiddenChart[y-1][x] ==9:
            hiddenChart[y-1][x] = 0
            replacedBombs += 1
        if y - 1 >= 0 and x + 1 < TILE_X_AMOUNT and hiddenChart[y-1][x+1] == 9:
            hiddenChart[y-1][x+1] = 0
            replacedBombs += 1
        if x - 1 >= 0 and hiddenChart[y][x-1] == 9:
            hiddenChart[y][x-1] = 0
            replacedBombs += 1
        if hiddenChart[y][x] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if x + 1 < TILE_X_AMOUNT and hiddenChart[y][x+1] == 9:
            hiddenChart[y][x+1] = 0
            replacedBombs += 1
        if y + 1 < TILE_Y_AMOUNT and x - 1 >= 0 and hiddenChart[y+1][x-1] == 9:
            hiddenChart[y+1][x-1] = 0
            replacedBombs += 1
        if y + 1 < TILE_Y_AMOUNT and hiddenChart[y+1][x] == 9:
            hiddenChart[y+1][x] = 0
            replacedBombs += 1
        if y + 1 < TILE_Y_AMOUNT and x + 1 < TILE_X_AMOUNT and hiddenChart[y+1][x+1] == 9:
            hiddenChart[y+1][x+1] = 0
            replacedBombs += 1
        #Get all cover tiles outside safety square
        safeTiles = []
        for i in range(0,TILE_Y_AMOUNT):
            for j in range (0,TILE_X_AMOUNT):
                if abs(i - y) <= 1 or abs(j - x) <= 1 or hiddenChart[i][j] == 9:
                    continue
                safeTiles.insert(0,(j,i))
        #Place replaced bombs
        while replacedBombs >0:
            randomNum = random.randint(0,len(safeTiles)- 1)
            newX, newY =  safeTiles[randomNum]
            hiddenChart[newY][newX] = 9
            replacedBombs -= 1
                
        #Place numbers near bombs
        for i in range(0,TILE_Y_AMOUNT):
            for j in range (0,TILE_X_AMOUNT):
                if hiddenChart[i][j] != 9:
                    continue
                if i - 1 >= 0 and j - 1 >= 0 and hiddenChart[i-1][j-1] != 9:
                    hiddenChart[i-1][j-1] += 1
                if i - 1 >= 0 and hiddenChart[i-1][j] !=9:
                    hiddenChart[i-1][j] += 1
                if i - 1 >= 0 and j + 1 < TILE_X_AMOUNT and hiddenChart[i-1][j+1] != 9:
                    hiddenChart[i-1][j+1] += 1
                if j - 1 >= 0 and hiddenChart[i][j-1] != 9:
                    hiddenChart[i][j-1] += 1
                if j + 1 < TILE_X_AMOUNT and hiddenChart[i][j+1] != 9:
                    hiddenChart[i][j+1] += 1
                if i + 1 < TILE_Y_AMOUNT and j - 1 >= 0 and hiddenChart[i+1][j-1] != 9:
                    hiddenChart[i+1][j-1] += 1
                if i + 1 < TILE_Y_AMOUNT and hiddenChart[i+1][j] !=9:
                    hiddenChart[i+1][j] += 1
                if i + 1 < TILE_Y_AMOUNT and j + 1 < TILE_X_AMOUNT and hiddenChart[i+1][j+1] != 9:
                    hiddenChart[i+1][j+1] += 1
    #automaticly uncover tiles that are not next to bombs        
    if hiddenChart[y][x] == 0:
        automaticUncover(x,y)
    else:
        score += 1
    firstMove = False

def placeBombs(amount):
    placedBombs = 0
    while placedBombs != amount:
        while True:
            x = random.randint(0,TILE_X_AMOUNT - 1)
            y = random.randint(0,TILE_Y_AMOUNT - 1)
            if hiddenChart[y][x] != 9:
                hiddenChart[y][x] = 9
                placedBombs += 1
                break

placeBombs(BOMB_AMOUNT)

#Create cover tiles
for i in range(0,TILE_Y_AMOUNT):
    for j in range (0,TILE_X_AMOUNT):
        if j%2==0:
            if i%2==0:
                chart[i][j] = -1
            else:
                chart[i][j] = -2
        else:
            if i%2==0:
                chart[i][j] = -2
            else:
                chart[i][j] = -1

        grid[i][j] = pygame.Rect(j * 32,i*32,32,32).copy()
    print("\n",end="")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx,my = pygame.mouse.get_pos()
            mx = int(mx/32)
            my = int(my/32)
            pickTile(mx,my)
            chart[my][mx] = hiddenChart[my][mx]          
            
            print(score,"/",SAFE_TILES)
            if hiddenChart[my][mx] == 9:
                revealBombs()
                print("You lost!")
                #Game lost terminate episode
            if score == SAFE_TILES:
                print("You won!")
                #Game won terminate episode


    #RENDER LOOP
    screen.fill((0,0,0))

    for i in range(0,TILE_Y_AMOUNT):
        for j in range (0,TILE_X_AMOUNT):
            drawTile(j,i,chart[i][j])         
 
    pygame.display.update()
    clock.tick(60) #60 framerate cap


#SOurces
#https://www.youtube.com/watch?v=M6e3_8LHc7A&list=WL&index=84&t=662s getSprite
#https://www.youtube.com/watch?v=AY9MnQ4x3zk&t=8207s main
#https://www.pygame.org/docs/
#https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/