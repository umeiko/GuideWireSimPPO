import numpy as np
# import collections
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from rl_agents import ppo
# from net.deeplabv3_plus import DeepLabEncoder

class example_network(ppo.BaseNetwork):
    def __init__(self, input_dense, output_dense):
        super().__init__()
        self.epoch:torch.Tensor = torch.tensor([0], dtype=torch.int32)
        self.h1 = nn.Linear(input_dense, 256)
        self.h2 = nn.Linear(256, 256)

        self.h1_C = nn.Linear(input_dense, 256)
        self.h2_C = nn.Linear(256, 256)

        self.h3 = nn.Linear(256, output_dense)
        self.h4 = nn.Linear(256, 1)

    def forward(self, x_in):
        x = F.relu(self.h1(x_in))
        x = F.relu(self.h2(x))
        out_a = F.softmax(self.h3(x), dim=1)

        x_ = F.relu(self.h1_C(x_in))
        x_ = F.relu(self.h2_C(x))
        out_c = self.h4(x_)
        
        return out_a, out_c

class CNN_FC(nn.Module):
    def __init__(self, input_channels=1, act_num=5, softmax=True):
        super().__init__()
        self.input_shape = (input_channels, 256, 256)
        self.softmax = softmax
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.cov_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 1024),
        )
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, act_num)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x:torch.Tensor):
        for k, shape in enumerate(self.input_shape):
            if x.shape[k+1] != shape:
                raise ValueError(f"Input shape should be {self.input_shape}, got {x.shape[1:]}")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.cov_out(x))
        if self.softmax:
            a = F.softmax(self.actor_linear(x), dim=1)
        else:
            a = self.actor_linear(x)
        return a, self.critic_linear(x)
    

# class DeepLab_FC(nn.Module):
#     def __init__(self, backbone, assp, act_num=5,device='cpu',input_channels=3, softmax=True):
#         super(DeepLab_FC, self).__init__()
#         self.input_shape = (1,input_channels, 512, 512)
#         # 加载Deeplab的Encoder部分
#         self.backbone= backbone
#         self.assp = assp
#         self.softmax = softmax

#         #冻结backbone和assp的参数
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         for param in self.assp.parameters():
#             param.requires_grad = False

#         # 预计算全连接层的输入特征数，
#         # demo_input = torch.zeros(1, 3, 512, 512)
#         demo_input = torch.zeros(1, 3, 700, 600)
#         demo_input = demo_input.to(device)
#         with torch.no_grad():
#             features = self.assp(self.backbone(demo_input)[0])
#         n_features = torch.numel(features) // features.shape[0]
#         self.fc = nn.Linear(n_features, 1024)
#         self.actor_linear = nn.Linear(1024, act_num)
#         self.critic_linear = nn.Linear(1024, 1)
#         self._initialize_weights()

#         # self.critic_linear = nn.Linear(1024, 1)
#         # self.actor_linear = nn.Linear(1024, act_num)
#         # self._initialize_weights()

#     def forward(self, x):
#         # for k, shape in enumerate(self.input_shape):
#         #     if x.shape[k] != shape:
#         #         raise ValueError(f"Input shape is {x.shape}, Expected {self.input_shape}")
#         # print("x",x.shape)
#         # 1. Encoder
#         x = x.permute(0, 3, 1, 2) 
#         # print("x0",x.shape)
#         x, low_level_features = self.backbone(x)
#         x = self.assp(x)
#         # print("x1",x.shape)
#         # 2. 全连接输出
#         # x = F.relu(self.cov_out(x))
#         # 动态调整全连接层
#         # _features = torch.numel(x) // x.shape[0] # 计算除了batch_size以外的所有特征数量
#         # print("n_features",n_features)
#         # self.fc = nn.Linear(n_features, 1024).to(x.device)

#         x = x.reshape(x.size(0), -1)  # 动态重塑
#         x = F.relu(self.fc(x))
#         # print("x2",x.shape)

#         # 3. 对输出进行处理
#         if self.softmax:
#             a = F.softmax(self.actor_linear(x), dim=1)
#         else:
#             a = self.actor_linear(x)
#         # print("x3,x4",a.shape,self.critic_linear(x).shape)
#         # print("finish 1 ")
#         return a, self.critic_linear(x)
    

#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
#                 nn.init.constant_(module.bias, 0)
    

# class DeepLab_FC(nn.Module):
#     def __init__(self, backbone, assp, act_num=5, device='cpu', softmax=True):
#         super(DeepLab_FC, self).__init__()
#         self.backbone = backbone
#         self.assp = assp
#         self.softmax = softmax
#         self.device = device

#         # 冻结backbone和assp的参数
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         for param in self.assp.parameters():
#             param.requires_grad = False

#         # 初始化全连接层为空，随后在第一次前向传播时定义
#         self.fc = None
#         self.actor_linear = nn.Linear(1024, act_num).to(self.device)
#         self.critic_linear = nn.Linear(1024, 1).to(self.device)
#         self.initialize_weights()

#     def forward(self, x):
#         print("x0",x.shape)
#         x = x.permute(0, 3, 1, 2)  # BxHxWxC to BxCxHxW
#         print("x1",x.shape)
#         x, features = self.backbone(x)
#         x = self.assp(x)
#         print("x2",x.shape)
   
#         # 动态计算特征数并创建全连接层
#         n_features = features.reshape(x.size(0), -1).size(1) 
#         self.fc = nn.Linear(n_features, 1024).to(self.device)
#         x = features.reshape(features.size(0), -1) 
#         x = F.relu(self.fc(x))
#         print("x3",x.shape)
#         if self.softmax:
#             action = F.softmax(self.actor_linear(x), dim=1)
#         else:
#             action = self.actor_linear(x)
#         value = self.critic_linear(x)
#         print("x4",action.shape)
#         print("x5",value.shape)
#         return action, value

#     def initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
#                 nn.init.constant_(module.bias, 0)
    

if __name__ == '__main__':
    ...
    # from torchsummary import summary
    # a = torch.rand([1, 3, 96, 96]).to("cuda:0")
    # model = Mario(act_num=6).to("cuda:0")
    # print(model(a))
    # summary(model, (3, 96, 96))