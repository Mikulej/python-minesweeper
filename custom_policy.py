import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
#Shared
class NoChangeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim: int = 256):
        super().__init__(spaces.Box(low=-1,high=9,shape=(16, 16),dtype=np.int8), features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations
    
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
        self.mlp_extractor = CustomNetwork(self.features_dim)
#Network #1 
class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 4,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

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
        return self.forward_actor(features), self.forward_critic(features)

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

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        x = features
        x = th.unsqueeze(x,1)
        x = F.relu(self.vf1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.vf2(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.vf3(x))
        x = th.flatten(x,start_dim=1)
        x = F.relu(self.vf4(x))
        return x

    



#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#default-network-architecture