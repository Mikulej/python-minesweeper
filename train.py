import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper
#from stable_baselines3 import A2C
from stable_baselines3 import DQN
import time

env = MineSweeper(renderMode="human")
observation, info = env.reset()
print("Observation space is:", env.observation_space)
print("Action space is:", env.action_space)
print("Action_masks is:",env.action_masks().__len__())
print("Possible actions: ",env.possible_actions)
print("Invalid actions: ",env.invalid_actions)
for _ in range(1000):
    time.sleep(1)

    #invalidActions = env.get_wrapper_attr('action_masks')
    print("Invalid actions are: ",env.invalid_actions)

    masking = env.action_masks()
    #print("Action masking is:", masking)
    #action, states = model.predict(observation,action_masks=invalidActions)
    action = env.action_space.sample()#policy goes here, sample is a purely random action
    
    print("Action is ",action) 
    observation, reward, terminated,truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated or truncated:
      observation, info = env.reset()
      print(info)
env.close()




#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html