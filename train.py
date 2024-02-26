import os
import pygame as pg
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from environment import Environment
from brain import Brain
from DQN import Dqn

# 設定參數
memSize = 60000         # 記憶容量上限
batchSize = 32          # 批次大小
learningRate = 0.0001   # 學習速率
gamma = 0.9             # 折扣率
nLastStates = 4         # 短期記憶中的步驟數

def main():
    # 初始化環境、大腦和 DQN，遊戲視窗設定小一點
    env = Environment(0, 100, 100)

    epsilon = 1. # 探索機率
    epsilonDecayRate = 0.0002 # 探索衰減率
    minEpsilon = 0.03 # 最低探索機率

    filepathToSave = f'model.pth'

    brain = Brain((nLastStates, env.nRows, env.nColumns), learningRate)

    # 檢查是否載入先前訓練的模型
    if os.path.exists(filepathToSave):
        load_model = input('是否要載入之前已經訓練的模型，再繼續訓練(y/n): ')
        if load_model.lower() == 'y':
            brain.loadModel('model.pth')
            epsilon = 0.5 # 隨機漫步率

    dqn = Dqn(memSize, gamma)

    # 重置遊戲狀態函數
    def resetStates():
        currentState = np.zeros((1, nLastStates, env.nRows, env.nColumns))
        for i in range(nLastStates):
            currentState[:, i, :, :] = env.screenMap
        return currentState, currentState

    # 訓練循環
    epoch = 0
    scores = []
    maxNCollected = 0
    nCollected = 0
    totNCollected = 0

    while True:
        # 重置環境和遊戲狀態
        env.reset()
        currentState, nextState = resetStates()
        epoch += 1
        gameOver = False

        while not gameOver:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
            else:
                with torch.no_grad():
                    current_state_tensor = torch.tensor(currentState, dtype=torch.float)
                    qvalues = brain(current_state_tensor)
                    action = torch.argmax(qvalues, dim=1).item()

            # 更新環境狀態
            state, reward, gameOver = env.step(action)

            # 更新下一狀態，添加新遊戲畫面到下一個狀態並刪除最舊的畫面
            state = np.reshape(state, (1, env.nRows, env.nColumns)) # 調整 state 的形狀為 (1, 10, 10)
            state = np.expand_dims(state, axis=1) # 調整為四維數組 (1, 1, 10, 10)
            nextState = np.append(nextState[:, 1:, :, :], state, axis=1) # 將 state 添加到 nextState
            
            # 記憶並訓練AI
            dqn.remember((currentState, action, reward, nextState), gameOver)
            inputs, targets = dqn.get_batch(brain, batchSize)

            inputs_tensor = torch.tensor(inputs, dtype=torch.float)
            targets_tensor = torch.tensor(targets, dtype=torch.float)

            brain.optimizer.zero_grad()
            predictions = brain(inputs_tensor)
            loss = brain.loss(predictions, targets_tensor)
            loss.backward()
            brain.optimizer.step()

            # 檢查是否收集蘋果並更新當前狀態
            if env.collected:
                nCollected += 1

            currentState = nextState

            # 使遊戲畫面更新
            pg.display.flip()

        # 保存模型如果打破記錄
        if nCollected > maxNCollected and nCollected > 2:
            maxNCollected = nCollected
            torch.save(brain.state_dict(), filepathToSave)

        totNCollected += nCollected
        nCollected = 0

        # 顯示每輪結果
        if epoch != 0:
            scores.append(totNCollected)
            epochs = list(range(len(scores))) # 創建與 scores 長度相同的 epoch 列表
            plt.scatter(epochs, scores) # 使用點圖顯示每輪分數
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # 設定 Y 軸刻度為整數
            plt.savefig('train_progress.png')
            plt.close()
            # 重置累計蘋果數量
            totNCollected = 0

        # 降低探索機率
        if epsilon > minEpsilon:
            epsilon -= epsilonDecayRate

        # 每輪遊戲結果顯示
        print('Epoch: ' + str(epoch) + ' Current Best: ' + str(maxNCollected) + ' Epsilon: {:.5f}'.format(epsilon))

if __name__ == "__main__":
    main()