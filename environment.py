import numpy as np
import pygame as pg

# 初始化環境類別
class Environment():
    def __init__(self, waitTime, width=480, height=480):
        # 定義參數
        self.width = width           # 遊戲視窗的寬度
        self.height = height         # 遊戲視窗的高度
        self.nRows = 10              # 棋盤的行數
        self.nColumns = 10           # 棋盤的列數
        self.initSnakeLen = 2        # 蛇的初始長度
        self.defReward = -0.03       # 執行動作的獎勵 - 生存懲罰
        self.negReward = -10.        # 死亡的獎勵
        self.posReward = 4.          # 收集蘋果的獎勵
        self.waitTime = waitTime     # 執行動作後的延遲時間

        # 如果蛇的初始長度大於棋盤行數的一半，則調整為行數的一半
        if self.initSnakeLen > self.nRows / 2:
            self.initSnakeLen = int(self.nRows / 2)

        # 設定遊戲視窗的模式
        self.screen = pg.display.set_mode((self.width, self.height))
        self.snakePos = list()

        # 創建一個數學表示遊戲棋盤的數組
        self.screenMap = np.zeros((self.nRows, self.nColumns))

        # 初始化蛇的位置: 數組操作
        self.snakePos = [(int(self.nRows / 2), int(self.nColumns / 4) + i) for i in range(self.initSnakeLen)]
        row_start = int(self.nRows / 2)
        self.screenMap[row_start:row_start + self.initSnakeLen, int(self.nColumns / 4)] = 0.5

        # 放置蘋果
        self.applePos = self.placeApple()
        
        # 繪製遊戲畫面
        self.drawScreen()

        # 設置是否收集蘋果的標誌和上一次移動的變數
        self.collected = False
        self.lastMove = 2

    # 建立一個方法來獲得蘋果的新隨機位置
    def placeApple(self):
        # 找出所有空的位置（即未被蛇佔用的位置）
        empty_positions = np.argwhere(self.screenMap != 0.5)

        # 從這些空位置中隨機選擇一個
        posy, posx = empty_positions[np.random.choice(len(empty_positions))]

        # 將選中的位置設置為 1，代表蘋果
        self.screenMap[posy][posx] = 1

        return (posy, posx)
    
    # 製作一個功能來繪製我們看到的所有東西
    def drawScreen(self):
        self.screen.fill((0, 0, 0))  # 使用黑色填充整個遊戲畫面

        cellWidth = self.width / self.nColumns  # 計算每個格子的寬度
        cellHeight = self.height / self.nRows    # 計算每個格子的高度
        snake_length = len(self.snakePos)
        # 遍歷每個格子並根據數據繪製蛇和蘋果
        for i in range(self.nRows):
            for j in range(self.nColumns):
                # 如果格子中的值為 0.5，表示這是蛇的一部分，用白色繪製
                if self.screenMap[i][j] == 0.5:
                    pg.draw.rect(self.screen, (140, 220, 60), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                # 如果格子中的值為 1，表示這是蘋果，用紅色繪製
                elif self.screenMap[i][j] == 1:
                    pg.draw.rect(self.screen, (255, 0, 0), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                        
        # pg.display.flip()  # 更新整個遊戲畫面，顯示繪製的結果

    # 更新蛇位置的方法
    def moveSnake(self, nextPos, col):
        self.snakePos.insert(0, nextPos)

        if not col:
            self.snakePos.pop()
        
        # 重置屏幕地圖，用零填充
        self.screenMap = np.zeros((self.nRows, self.nColumns))

        # 使用 NumPy 快速更新蛇在屏幕地圖上的位置
        snake_positions = np.array(self.snakePos)
        self.screenMap[snake_positions[:, 0], snake_positions[:, 1]] = 0.5

        # 如果蛇吃到了蘋果，放置新的蘋果
        if col:
            self.applePos = self.placeApple()
            self.collected = True

        # 將蘋果在屏幕地圖上的位置標記為1（紅色）
        self.screenMap[self.applePos] = 1
    
    # 主要方法來更新環境
    def step(self, action):
        # 重設這些參數並將獎勵設為生存懲罰
        gameOver = False
        reward = self.defReward
        self.collected = False

        # 確保動作不與上一次動作相反
        action = self.correct_action(action)

        # 更新蛇頭座標
        snakeX, snakeY = self.update_snake_head(action)

        # 檢查蛇頭位置
        if self.is_collision(snakeX, snakeY):
            return self.screenMap, self.negReward, True  # 返回負獎勵和遊戲結束標誌

        # 檢查是否吃到蘋果
        ate_apple = self.screenMap[snakeY][snakeX] == 1
        reward = self.posReward if ate_apple else self.defReward

        # 更新蛇的位置
        self.moveSnake((snakeY, snakeX), ate_apple)

        # 繪製屏幕
        self.drawScreen()

        # 更新最後一次動作
        self.lastMove = action

        # 等待指定時間
        pg.time.wait(self.waitTime)

        return self.screenMap, reward, False

    def correct_action(self, action):
        # 防止蛇向相反方向移動
        opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        return action if action != opposite_actions.get(self.lastMove, -1) else self.lastMove

    def update_snake_head(self, action):
        # 根據動作更新蛇頭座標
        # action = 0 -> 向上
        # action = 1 -> 向下
        # action = 2 -> 向右
        # action = 3 -> 向左
        snakeX, snakeY = self.snakePos[0][1], self.snakePos[0][0]
        if action == 0: snakeY -= 1
        elif action == 1: snakeY += 1
        elif action == 2: snakeX += 1
        elif action == 3: snakeX -= 1
        return snakeX, snakeY

    def is_collision(self, x, y):
        # 檢查蛇頭是否碰到邊界或自己
        return (x < 0 or x >= self.nColumns or y < 0 or y >= self.nRows or
                self.screenMap[y][x] == 0.5)

    
    # 重置遊戲環境的函數
    def reset(self):
        self.screenMap  = np.zeros((self.nRows, self.nColumns))  # 初始化屏幕地圖為全零，表示空白區域
        self.snakePos = list()  # 初始化蛇的位置列表為空列表
        
        # 在屏幕中心位置初始化蛇的初始長度
        self.snakePos = [(int(self.nRows / 2), int(self.nColumns / 4) + i) for i in range(self.initSnakeLen)]
        row_start = int(self.nRows / 2)
        self.screenMap[row_start:row_start + self.initSnakeLen, int(self.nColumns / 4)] = 0.5
        
        # 放置新的蘋果
        self.applePos = self.placeApple()
        self.screenMap[self.applePos[0], self.applePos[1]] = 1
        
        self.lastMove = 2  # 重置最後一次移動的方向為


# 啟用我們自己玩遊戲的功能，如果我們執行這個 "environment.py" 檔案
if __name__ == '__main__':
    env = Environment(100)
    start = True
    direction = 2
    gameOver = False
    reward = 0
    while True:
        state = env.screenMap
        pos = env.snakePos

        for event in pg.event.get():
            if event.type == pg.QUIT:
                gameOver = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP and direction != 1:
                    direction = 0
                elif event.key == pg.K_RIGHT and direction != 3:
                    direction = 2
                elif event.key == pg.K_LEFT and direction != 2:
                    direction = 3
                elif event.key == pg.K_DOWN and direction != 0:
                    direction = 1

        if start:
            screenMap, reward, gameOver = env.step(direction)
            # 檢視輸出結果
            # print('-'*50)
            # print(type(screenMap))
            # print(screenMap.shape)
            # print(screenMap)
            # print('reward: ', reward)
        if gameOver:
            env.reset()
            direction = 2

        # 使遊戲畫面更新
        pg.display.flip()