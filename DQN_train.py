from stable_baselines3 import DQN
from enviornement import MineSweeper
from train import evaluate
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from enviornement import MineSweeper
from train import evaluate

from sb3_contrib import MaskablePPO
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
# from sb3_contrib.common.maskable.utils import get_action_masks
# from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
# from stable_baselines3.common.monitor import Monitor

import time
import torch as th
import pickle
import matplotlib.pyplot as plt

import torch as th
import torch.nn as nn

from custom_policy import CustomDQN
from custom_policy import NoChangeExtractor
#Train Model
env = MineSweeper(render_mode=None,sizeX=16,sizeY=16,bombs=40)
if env == None:
    print("Error: Failed to create an enviornment")
    
#Network #1
observation, info = env.reset()
policy_kwargs = dict(
    # net_arch= [256,64,32],
    # activation_fn=th.nn.ReLU,
    #features_extractor_class=NoChangeExtractor,
   #features_extractor_kwargs=dict(features_dim=env.TILE_X_AMOUNT*env.TILE_Y_AMOUNT),
   features_extractor = nn.Flatten(),
   features_dim=env.TILE_X_AMOUNT*env.TILE_Y_AMOUNT,

)

#model = DQN("MlpPolicy", env,policy_kwargs=policy_kwargs)
model = DQN(CustomDQN, env,policy_kwargs=policy_kwargs)


performence_arr = []
iterations_arr = []
plt.subplots(figsize=(10, 5))

print("Learning...")
steps = 10000
for i in range(1,50):
    start_time = time.time()
    print("step=",i)
    model.learn(total_timesteps=steps,reset_num_timesteps=False,tb_log_name="DQN")
    model.save(f"models/{steps*i}")
    print("Evaluating policy...")
    performence = evaluate(model,env,timesteps=500)
    #performence = evaluate_random(model,env,timesteps=500)
    stop_time = time.time()
    print("Mean score: ",performence[0]," Win-rate: ",performence[1],"%")
    print("--- %s seconds ---" % (stop_time - start_time))

    performence_arr.append(performence[0])
    iterations_arr.append(steps*i)
    with open('total_performence.pkl', 'wb') as handle:
        pickle.dump(performence_arr, handle)
    with open('iterations_arr.pkl', 'wb') as handle:
        pickle.dump(iterations_arr, handle)

    plt.plot(iterations_arr,performence_arr)
    plt.xlabel('Iteration')
    plt.ylabel('Mean score')
    plt.savefig('score_plot.png')

print("Learning finished.")

#Test Model
print("Playing...")
env = MineSweeper(render_mode="human",sizeX=16,sizeY=16,bombs=40)
performence = evaluate(model,env,timesteps=500)
print("Mean score: ",performence[0]," Win-rate: ",performence[1],"%")

env.close()          
   

#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py
#https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py
        
