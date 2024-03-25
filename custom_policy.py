import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn.functional as F

# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = th.nn.Sequential(
#             th.nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=4, padding=0),
#             # th.nn.ReLU(),
#             # th.nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=0),
#             th.nn.ReLU(),
#             th.nn.Flatten(),
#         )

#         #Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]
#         self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.ReLU())
#         #self.linear = th.nn.Sequential(th.nn.Linear(n_flatten, features_dim), th.nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))

class MyNetwork(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(spaces.Box, features_dim)
        #self.l0 = nn.Linear(features_dim,2)
        # self.l1 = nn.Linear(features_dim,256)
        # self.l2 = nn.Linear(256,128)
        # self.l3 = nn.Linear(128,64)
        self.l1 = nn.Linear(features_dim,16)# A
        self.l2 = nn.Linear(16,8)
        self.l3 = nn.Linear(8,4) # B



        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = th.flatten(observations)
        observations = F.relu(self.l1(observations))
        observations = F.relu(self.l2(observations))
        observations = F.relu(self.l3(observations))
        print("Shape is ", observations)
        return observations
#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#default-network-architecture