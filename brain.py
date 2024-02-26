import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Brain(nn.Module):
    def __init__(self, iS=(1, 10, 10), lr=0.0005):  # 初始化函數 (通道(一次輸入多長的短期回憶), 長度, 寬度)
        super(Brain, self).__init__()
        # 定義第一個卷積層，輸入通道為 iS[0]
        self.conv1 = nn.Conv2d(in_channels=iS[0], out_channels=32, kernel_size=3)
        # 定義第二個卷積層
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        # 定義第一個全連接層
        self.fc1 = nn.Linear(in_features=self.calculate_conv_output_size(iS), out_features=256)
        # 定義第二個全連接層
        self.fc2 = nn.Linear(in_features=256, out_features=4)
        # 定義優化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # 定義損失函數
        self.loss = nn.MSELoss()


    def calculate_conv_output_size(self, iS):
        # 計算卷積層輸出的大小
        sample = torch.zeros(1, *iS)
        sample = F.relu(self.conv1(sample))
        sample = F.max_pool2d(sample, (2, 2))
        sample = F.relu(self.conv2(sample))
        return int(torch.numel(sample) / sample.size(0))

    def forward(self, x):
        # 前向傳播函數
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def loadModel(self, filepath):
        # 加載模型
        self.load_state_dict(torch.load(filepath))
        self.eval()

# 测试代码
if __name__ == '__main__':
    # 創建一個示例遊戲地圖，此處是 10x10 的二維數組
    screenMap = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])

    # 轉換為 PyTorch 張量並增加通道維度 (10, 10) -> (1, 1, 10, 10)
    # = (batch_size, channels, height, width)
    screenMaps_tensor = torch.tensor(screenMap, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    # 創建 Brain 實例
    brain = Brain((1, 10, 10), lr=0.01)

    with torch.no_grad():
        # 不計算梯度，進行推理
        qvalues = brain(screenMaps_tensor)
        actions = torch.argmax(qvalues, dim=1)

    # 打印輸出
    print("Output:", qvalues.numpy())
    print("Output Shape:", qvalues.shape)
    print("Action:", actions)
