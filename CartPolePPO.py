import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
#from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
env = gym.make('CartPole-v1',render_mode="human")

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones,tr, info = env.step(action)
    env.render()
    if dones:
        print("Reset")
        env.reset()