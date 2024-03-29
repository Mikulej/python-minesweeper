import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn.functional as F


# class MyNetwork(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         self.l1 = nn.Linear(features_dim,16)
#         self.l2 = nn.Linear(16,8)
#         self.l3 = nn.Linear(8,features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         observations = th.flatten(observations,start_dim=1)
#         observations = F.relu(self.l1(observations))
#         observations = F.relu(self.l2(observations))
#         observations = F.relu(self.l3(observations))
#         return observations
#Convolution Network #0
# class MyNetwork(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         self.l1 = nn.Conv2d(1, 1, kernel_size=(5,5), stride=1, padding=2,padding_mode='zeros')
#         self.l2 = nn.Linear(features_dim,features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         #observations = th.flatten(observations,start_dim=1)
#         observations = th.unsqueeze(observations,1)
#         observations = F.relu(self.l1(observations))
#         observations = th.flatten(observations,start_dim=1)
#         observations = F.relu(self.l2(observations))
#         return observations
#Convolution Network #1
# class MyNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         self.l1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=1, padding=2,padding_mode='zeros')
#         self.l2 = nn.Conv2d(64, 256, kernel_size=(5,5), stride=1, padding=2,padding_mode='zeros')
#         self.l3 = nn.Conv2d(256, 1, kernel_size=(5,5), stride=1, padding=2,padding_mode='zeros')
#         self.l4 = nn.Linear(features_dim,features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         observations = th.unsqueeze(observations,1)
#         observations = F.relu(self.l1(observations))
#         observations = F.relu(self.l2(observations))
#         observations = F.relu(self.l3(observations))
#         observations = th.flatten(observations,start_dim=1)
#         observations = F.relu(self.l4(observations))
#         return observations
#Convolution Network #2
class MyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
        super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.l4 = nn.Linear(2048,features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        x = th.unsqueeze(x,1)
        x = F.relu(self.l1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.l2(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.l3(x))
        x = th.flatten(x,start_dim=1)
        x = F.relu(self.l4(x))
        return x

#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#default-network-architecture