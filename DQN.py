import numpy as np
import torch
from brain import Brain
from environment import Environment

"""
DQN（Deep Q-Network）演算法是一種結合傳統Q學習與深度學習的強化學習方法。DQN 主要用於解決具有高維度觀察空間的問題，例如視覺輸入的任務。
"""

class Dqn(object):
    """
    Return:
    inputs.shape = (batch, 4, 10, 10)  # 輸入的形狀：批次、通道、高度、寬度
    targets.shape = (batch, 4)         # 目標的形狀：批次、輸出值
    """
    def __init__(self, max_memory=100, discount=0.9):
        self.memory = list()             # 初始化記憶列表
        self.max_memory = max_memory     # 記憶的最大數量
        self.discount = discount         # 折扣因子

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])  # 將轉換和遊戲狀態添加到記憶中
        if len(self.memory) > self.max_memory:       # 如果記憶超過最大限制，刪除最舊的記憶
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        # 獲得批次的記憶，用於訓練模型
        len_memory = len(self.memory)
        num_outputs = model.fc2.out_features

        inputs = np.zeros((min(len_memory, batch_size), self.memory[0][0][0].shape[1],self.memory[0][0][0].shape[2],self.memory[0][0][0].shape[3]))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            
            with torch.no_grad():
                inputs[i] = current_state
                current_state_tensor = torch.tensor(current_state, dtype=torch.float)
                targets[i] = model(current_state_tensor).numpy()[0]
                next_state_tensor = torch.tensor(next_state, dtype=torch.float)
                Q_sa = np.max(model(next_state_tensor).numpy()[0])

            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
                
        return inputs, targets


if __name__ == "__main__":
    
    env = Environment(10)
    nLastStates = 4 # 4步的短期回憶
    model = Brain((nLastStates, env.nRows, env.nColumns), lr=0.01) # (4步的短期回憶, 高度, 寬度)
    dqn = Dqn(max_memory=100, discount=0.9)
    batchSize = 10

    def resetStates():
        currentState = np.zeros((1, nLastStates, env.nRows, env.nColumns)) # 初始化當前狀態
        
        for i in range(nLastStates):
            currentState[:,i,:,:] = env.screenMap # (batch_size, channels, height, width)
        
        return currentState, currentState

    # 創建一些 memory 用於測試
    currentState, nextState = resetStates()

    for _ in range(50):
        current_state_tensor = torch.tensor(currentState, dtype=torch.float)

        game_over = np.random.choice([True, False])
        qvalues = model(current_state_tensor)
        action = torch.argmax(qvalues, dim=1) # 選擇動作

        state, reward, gameOver = env.step(action) # 更新環境
        state = np.reshape(state, (1, 1, env.nRows, env.nColumns))
        nextState = np.append(nextState, state, axis = 1) # 更新下一狀態，儲存在 channel 的維度上
        nextState = np.delete(nextState, 0, axis = 1) # 刪除舊狀態，刪除 channel 的維度上

        dqn.remember([currentState, action, reward, nextState], gameOver) # 記憶當前轉換

        inputs, targets = dqn.get_batch(model, batchSize) # 獲取訓練批次
        inputs_tensor = torch.tensor(inputs, dtype=torch.float)
        targets_tensor = torch.tensor(targets, dtype=torch.float)

        model.optimizer.zero_grad()
        predictions = model(inputs_tensor) # 進行預測
        loss = model.loss(predictions, targets_tensor) # 計算損失
        loss.backward() # 反向傳播
        model.optimizer.step() # 更新權重
        print("Loss:", loss.item()) # 輸出損失

