import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
from enviornement import MineSweeper

env = MineSweeper()
observation = env.reset(seed=42)
for _ in range(500):
    action = env.action_space.sample()
    print("Action is ",action) 
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
      observation, info = env.reset()
env.close()

