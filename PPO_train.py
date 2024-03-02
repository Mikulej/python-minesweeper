from enviornement import MineSweeper
from train import evaluate

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

#Train Model
env = MineSweeper(render_mode="human",sizeX=16,sizeY=16,bombs=40)
#check_env(env)
observation, info = env.reset()
model = PPO("MlpPolicy", env,
                    learning_rate=0.0003,
                    n_steps= 2048,
                    batch_size= 64,
                    n_epochs= 10,
                    gamma= 0.95)
print("Learning...")
model.learn(total_timesteps=1_000)
print("Learning finished.")

#Test Model
print("Evaluating policy...")
performence = evaluate(model,env,timesteps=500)
print("Mean score: ",performence[0]," Mean game completation: ",performence[1],"%")
# performence = evaluate_policy(model,env)
# print("Mean reward=",performence[0]," Mean numbers of steps=",performence[1])

env.close()          
   
#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py