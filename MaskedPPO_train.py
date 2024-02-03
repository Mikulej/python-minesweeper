import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

#Train model
env = MineSweeper(renderMode="human")
#check_env(env)
observation, info = env.reset()
model = MaskablePPO("MlpPolicy", env, gamma=0.95, seed=32, verbose=1)
model.learn(total_timesteps=10000)

#Test Model
for _ in range(1000):
    
    invalidActions = env.action_masks()
    action, states = model.predict(observation,action_masks=invalidActions)
    observation, reward, terminated,truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated or truncated:
      observation, info = env.reset()
      print(info)
env.close()

#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py