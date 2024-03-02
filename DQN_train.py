from stable_baselines3 import DQN
from enviornement import MineSweeper
from train import evaluate
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

env = MineSweeper(render_mode="human",sizeX=16,sizeY=16,bombs=40)
#check_env(env)
observation, info = env.reset()
model = DQN("MlpPolicy", env)
print("Learning...")
model.learn(total_timesteps=10_000)
print("Learning finished.")

#Test Model
print("Evaluating policy...")
performence = evaluate(model,env,timesteps=500)
print("Mean score: ",performence[0]," Mean game completation: ",performence[1],"%")
#performence = evaluate_policy(model,env)
#print("Mean reward=",performence[0]," Mean numbers of steps=",performence[1])

env.close()  

        
