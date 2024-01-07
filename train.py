import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper

env = MineSweeper(renderMode="console")
check_env(env,warn=True)
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    print("Action is ",action) 
    observation, reward, terminated = env.step(action)
    env.render(env.RENDER_MODE)
    if terminated:
      print("Terminated with score: ",env.score,"/",env.WINNING_SCORE)
      observation = env.reset()
env.close()

