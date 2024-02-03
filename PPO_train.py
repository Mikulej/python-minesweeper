import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper
#from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO

#Train model
env = MineSweeper(renderMode="human")
check_env(env)
observation, info = env.reset()
model = PPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(total_timesteps=5000)

#Test Model
for _ in range(1000):
    
     #TO DO: Pick only legal moves
    invalidActions = env.get_action_masks()
    #invalidActions = env.get_wrapper_attr('action_masks')
    #print("Invalid actions are: ",invalidActions)
    action, states = model.predict(observation)
    #action = env.action_space.sample()
    
    #print("Action is ",action) 
    observation, reward, terminated,truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated or truncated:
      observation, info = env.reset()
      print(info)
env.close()

# for _ in range(1000):

#      #TO DO: Pick only legal moves
#     invalidActions = env.get_action_masks()
#     #invalidActions = env.get_wrapper_attr('action_masks')
#     print("Invalid actions are: ",invalidActions)
#     #action, states = model.predict(observation,action_masks=invalidActions)
#     action = env.action_space.sample()#policy goes here, sample is a purely random action
    
#     print("Action is ",action) 
#     observation, reward, terminated,truncated, info = env.step(action)
#     env.render(env.RENDER_MODE)
#     if terminated or truncated:
#       observation, info = env.reset()
#       print(info)
# env.close()
