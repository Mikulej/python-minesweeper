import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper
#from stable_baselines3 import A2C
from stable_baselines3 import DQN
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

env = MineSweeper(renderMode="human")
#model = MaskablePPO("MlpPolicy", env,gamma =0.4)
#model.learn(1000)
#evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=200, warn=False)
#model.save("PPO-MineSweeper")
#del model 
#model = MaskablePPO.load("PPO-MineSweeper")
#check_env(env,warn=True)
observation, info = env.reset()
for _ in range(1000):
    time.sleep(1)
     #TO DO: Pick only legal moves
    invalidActions = env.get_action_masks()
    print("Invalid actions are: ",invalidActions)
    action = env.action_space.sample()#policy goes here, sample is a purely random action
    
    print("Action is ",action) 
    observation, reward, terminated,truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated or truncated:
      observation, info = env.reset()
      print(info)
env.close()



