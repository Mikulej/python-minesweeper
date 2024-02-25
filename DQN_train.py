import gymnasium as gym

from stable_baselines3 import DQN
from enviornement import MineSweeper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

env = MineSweeper(renderMode="human",sizeX=16,sizeY=16,bombs=40)
#check_env(env)
observation, info = env.reset()
model = DQN("MlpPolicy", env)
print("Learning...")
model.learn(total_timesteps=100_000, log_interval=4)
print("Learning finished.")
#performence = evaluate_policy(model,env)
#print("Mean reward=",performence[0]," Mean numbers of steps=",performence[1])

#model.save("saperDQN")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_cartpole")

obs, info = env.reset()
for _ in range(5000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated or truncated:
        obs, info = env.reset()
        print(info)
        
