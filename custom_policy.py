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
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         print(observation_space)
#         n_input_channels = observation_space.shape[0]
#         print(n_input_channels)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         #UP TO THIS MOMENT SHOULD BE OK

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#         # self.linear = nn.Sequential(nn.Linear(self.cnn(
#         #         th.as_tensor(
#         #             self._observation_space.sample()[None]
#         #             ).float()
#         #         ),features_dim,nn.ReLU))

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         # print(
#         #     self.cnn(
#         #         th.as_tensor(
#         #             self._observation_space.sample()[None],device='cuda'
#         #             ).float()
#         #         )
#         # )
#         return self.linear(self.cnn(observations))


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
class MyNetwork(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
        super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
        self.l1 = nn.Conv2d(1, 1, kernel_size=(5,5), stride=1, padding=2,padding_mode='zeros')
        self.l2 = nn.Linear(features_dim,features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #observations = th.flatten(observations,start_dim=1)
        observations = th.unsqueeze(observations,1)
        observations = F.relu(self.l1(observations))
        observations = th.flatten(observations,start_dim=1)
        observations = F.relu(self.l2(observations))
        return observations

#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#default-network-architecture