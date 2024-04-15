import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

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
#Convolution Feature Extractor #0
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
#Convolution Feature Extractor  #1
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
#Convolution Feature Extractor #2
# class MyNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l4 = nn.Linear(2048,features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         x = observations
#         x = th.unsqueeze(x,1)
#         x = F.relu(self.l1(x))
#         x = F.max_pool2d(x,kernel_size=(2,2))
#         x = F.relu(self.l2(x))
#         x = F.max_pool2d(x,kernel_size=(2,2))
#         x = F.relu(self.l3(x))
#         x = th.flatten(x,start_dim=1)
#         x = F.relu(self.l4(x))
#         return x
#Convolution Feature Extractor  #3
# class MyNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
#         super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)
#         self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
#         self.l4 = nn.Linear(2048,features_dim)

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         x = observations
#         x = th.unsqueeze(x,1)
#         x = self.l1(x)
#         x = F.max_pool2d(x,kernel_size=(2,2))
#         x = self.l2(x)
#         x = F.max_pool2d(x,kernel_size=(2,2))
#         x = self.l3(x)
#         x = th.flatten(x,start_dim=1)
#         x = self.l4(x)
#         return x
#Convolution Network #4 (Real Neural Netwrok)
class NoChangeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
        super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations
    
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 4,
    ):
        super().__init__()
        #pi=actor
        #vf=critic
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        # self.policy_net = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros'),
        #     nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros'),
        #     nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros'),
        #     nn.Linear(2048, last_layer_dim_pi), nn.ReLU()
        # )

        self.pi1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.pi2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.pi3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.pi4 = nn.Linear(2048, last_layer_dim_pi)

        self.vf1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.vf2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.vf3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1,padding_mode='zeros')
        self.vf4 = nn.Linear(2048, last_layer_dim_vf)
        
        self.vftest = nn.Linear(feature_dim, last_layer_dim_vf)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #features = th.unsqueeze(features,1)
        return self.forward_actor(features), self.forward_critic(features)

    # def forward_actor(self, features: th.Tensor) -> th.Tensor:
    #     return self.policy_net(features)
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        x = features
        x = th.unsqueeze(x,1)
        x = F.relu(self.pi1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.pi2(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.pi3(x))
        x = th.flatten(x,start_dim=1)
        x = F.relu(self.pi4(x))
        return x

    # def forward_critic(self, features: th.Tensor) -> th.Tensor:
    #     return self.value_net(features)
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # x = features
        # x = th.unsqueeze(x,1)
        # x = F.relu(self.vf1(x))
        # x = F.max_pool2d(x,kernel_size=(2,2))
        # x = F.relu(self.vf2(x))
        # x = F.max_pool2d(x,kernel_size=(2,2))
        # x = F.relu(self.vf3(x))
        # x = th.flatten(x,start_dim=1)
        # x = F.relu(self.vf4(x))
        # return x
        x = features
        x = th.flatten(x,start_dim=1)
        x = F.relu(self.vftest(x))
        return x

    
class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        #print("feature_dim=", self.features_dim)
        self.mlp_extractor = CustomNetwork(self.features_dim)


#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#default-network-architecture