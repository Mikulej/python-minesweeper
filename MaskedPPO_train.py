from enviornement import MineSweeper
from train import evaluate_mask

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
#from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
#Train Model
env = MineSweeper(render_mode=None,sizeX=16,sizeY=16,bombs=40)
if env == None:
    print("Error: Failed to create an enviornment")
#check_env(env)
observation, info = env.reset()
model = MaskablePPO("MlpPolicy", env)
# model = MaskablePPO("MlpPolicy", env,
#                     learning_rate=0.0003,
#                     n_steps= 2048,
#                     batch_size= 64,
#                     n_epochs= 10,
#                     gamma= 0.4,policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256, 256]))
#,tensorboard_log="logs"
print("Learning...")
# steps = 1000
# for i in range(1,10):
#     model.learn(total_timesteps=steps,use_masking=True,reset_num_timesteps=False,tb_log_name="MaskPPO")
#     model.save(f"models/{steps*i}")
model.learn(total_timesteps=10000,use_masking=True)
print("Learning finished.")

#Test Model
print("Evaluating policy...")
performence = evaluate_mask(model,env,timesteps=500)
print("Mean score: ",performence[0]," Win-rate: ",performence[1],"%")
print("Playing...")
env = MineSweeper(render_mode="human",sizeX=16,sizeY=16,bombs=40)
performence = evaluate_mask(model,env,timesteps=500)
# performence = evaluate_policy(model,env,use_masking=True)
# print("Mean reward=",performence[0]," Mean numbers of steps=",performence[1])



env.close()          
   
#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py