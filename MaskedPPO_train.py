from enviornement import MineSweeper
from train import evaluate_mask

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
#from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

#Train Model
env = MineSweeper(renderMode="human",sizeX=16,sizeY=16,bombs=40)
#check_env(env)
observation, info = env.reset()
model = MaskablePPO("MlpPolicy", env,
                    learning_rate=0.0003,
                    n_steps= 2048,
                    batch_size= 64,
                    n_epochs= 10,
                    gamma= 0.99)
print("Learning...")
model.learn(total_timesteps=10_000,use_masking=True)
print("Learning finished.")

#Test Model
print("Evaluating policy...")
performence = evaluate_mask(model,env,timesteps=500)
print("Mean score: ",performence[0]," Mean game completation: ",performence[1],"%")
# performence = evaluate_policy(model,env,use_masking=True)
# print("Mean reward=",performence[0]," Mean numbers of steps=",performence[1])



env.close()          
   
#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py