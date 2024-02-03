import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import numpy as np

# def mask_fn(env: gym.Env) -> np.ndarray:
#     # Do whatever you'd like in this function to return the action mask
#     # for the current env. In this example, we assume the env has a
#     # helpful method we can rely on.
#     return env.get_action_masks()
#Train model
env = MineSweeper(renderMode="human")
#env = ActionMasker(env,env.action_masks())
check_env(env)
observation, info = env.reset()
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(total_timesteps=5000)

#Test Model
# for _ in range(1000):
    
#      #TO DO: Pick only legal moves
#     invalidActions = env.get_action_masks()
#     #invalidActions = env.get_wrapper_attr('action_masks')
#     #print("Invalid actions are: ",invalidActions)
#     action, states = model.predict(observation,action_masks=invalidActions)
#     #action = env.action_space.sample()
    
#     #print("Action is ",action) 
#     observation, reward, terminated,truncated, info = env.step(action)
#     env.render(env.RENDER_MODE)
#     if terminated or truncated:
#       observation, info = env.reset()
#       print(info)
# env.close()

#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py