import os
import pygame as pg
import torch
from environment import Environment
from brain import Brain
import numpy as np
from train import nLastStates

"""
使用訓練好的模型玩貪吃蛇，模型權重檔名為 model.pth，如果尚未訓練。
"""

# 函數重置遊戲狀態
def resetStates(screenMap):
    currentState = np.zeros((1, nLastStates, env.nRows, env.nColumns))
    for i in range(nLastStates):
        currentState[:, i, :, :] = screenMap
    return currentState, currentState

# 初始化環境
env = Environment(100, 480, 480)

# 初始化大腦
brain = Brain((nLastStates, env.nRows, env.nColumns), 0.0005)

# 載入模型
filepathToSave = 'model.pth'
loadSelfModel = input("是否載入預先訓練的模型(y/n): ")
if loadSelfModel.lower() == 'y':
    brain.loadModel('model_default.pth')
else:
    if os.path.exists(filepathToSave):
        brain.loadModel('model.pth')
    else:
        print("模型尚未訓練，請先執行 train.py。")
    


def play_game(brain, env):
    pg.font.init()  # 初始化字體模塊
    font = pg.font.SysFont('arial', 24)  # 選擇字體和大小

    # 遊戲初始化
    env.reset()
    currentState, nextState = resetStates(env.screenMap)
    gameOver = False
    score = 0  # 初始化分數

    while not gameOver:

        # 預測動作
        with torch.no_grad():
            current_state_tensor = torch.tensor(currentState, dtype=torch.float)
            qvalues = brain(current_state_tensor)
            action = torch.argmax(qvalues, dim=1).item()

        # 更新環境
        state, reward, gameOver = env.step(action)

        # 更新分數
        if env.collected:  # 如果蛇吃到了蘋果
            score += 1

        # 添加新遊戲畫面到下一個狀態並刪除最舊的畫面
        state = np.reshape(state, (1, env.nRows, env.nColumns))  # 將 state 調整為 (1, 10, 10)
        state = np.expand_dims(state, axis=1)  # 調整為四維數組，形狀為 (1, 1, 10, 10)
        nextState = np.append(nextState[:, 1:, :, :], state, axis=1)  # 將 state 添加到 nextState
        currentState = nextState

        # 繪製遊戲畫面
        env.drawScreen()

        # 繪製分數
        score_surface = font.render(f'Score: {score}', True, (255, 255, 255))  # 創建文字表面
        env.screen.blit(score_surface, (10, 10))  # 將文字繪製到螢幕上

        
        # 使遊戲畫面更新
        pg.display.flip()

# 玩 n 次遊戲
for _ in range(10):
    play_game(brain, env)
