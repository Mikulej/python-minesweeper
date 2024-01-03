import pygame
import random
import numpy as np
from sys import exit


pygame.init()
screen = pygame.display.set_mode((768,640))
TILE_X_AMOUNT = int(768 / 32)
TILE_Y_AMOUNT = int(640 / 32)
BOMB_AMOUNT = 200
pygame.display.set_caption('Minesweeper')
clock = pygame.time.Clock()
firstMove = True
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
def pickTile(x,y):
    global firstMove
    if firstMove == True:
        replacedBombs = 0
        if y - 1 >= 0 and x - 1 >= 0 and hiddenChart[y-1][x-1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if y - 1 >= 0 and hiddenChart[y-1][x] ==9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if y - 1 >= 0 and x + 1 < TILE_X_AMOUNT and hiddenChart[y-1][x+1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1

        if x - 1 >= 0 and hiddenChart[y][x-1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if hiddenChart[y][x] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if x + 1 < TILE_X_AMOUNT and hiddenChart[y][x+1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1

        if y + 1 < TILE_Y_AMOUNT and x - 1 >= 0 and hiddenChart[y+1][x-1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if y + 1 < TILE_Y_AMOUNT and hiddenChart[y+1][x] ==9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        if y + 1 < TILE_Y_AMOUNT and x + 1 < TILE_X_AMOUNT and hiddenChart[y+1][x+1] == 9:
            hiddenChart[y][x] = 0
            replacedBombs += 1
        #replace bombs outside of the safety square
        while replacedBombs > 0:
            while True:
                newX = 0
                newY = 0
                #TO DO better approach would be to map all reaming tiles and randomly choose one of them
                while abs(x - newX) <= 1:
                    newX = random.randint(0,TILE_X_AMOUNT - 1)
                while abs(y - newY) <= 1:
                    newY = random.randint(0,TILE_Y_AMOUNT - 1)
                if hiddenChart[newY][newX] != 9:
                    hiddenChart[newY][newX] = 9
                    replacedBombs -= 1
                    break
        #reassign numbers
        
    firstMove = False
    


    

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

chart = np.zeros((TILE_Y_AMOUNT,TILE_X_AMOUNT))
hiddenChart = np.zeros((TILE_Y_AMOUNT,TILE_X_AMOUNT))
grid = np.empty((TILE_Y_AMOUNT,TILE_X_AMOUNT),dtype=pygame.Rect)



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
            print(chart[i][j])
        
        grid[i][j] = pygame.Rect(j * 32,i*32,32,32).copy()
    print("\n",end="")

#Create content of tiles
#Place BOMB_AMOUNT bombs
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

    
peek()
#chart = hiddenChart
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
            
            
            if hiddenChart[my][mx] == 9:
                revealBombs()
                #terminate episode



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