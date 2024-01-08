import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper
#from stable_baselines3 import A2C
from stable_baselines3 import DQN
import time

env = MineSweeper(renderMode="human")
#check_env(env,warn=True)
observation = env.reset()
for _ in range(1000):
    #time.sleep(1)
    action = env.action_space.sample()
    print("Action is ",action) 
    observation, reward, terminated = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated:
      observation, info = env.reset()
      print(info)
env.close()


# env = MineSweeper(renderMode="console")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)
# model.save("saperBot")

# del model

# model = DQN.load("saperBot")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=False)
#     obs, reward, terminated = env.step(action)
#     if terminated:
#         obs = env.reset()
