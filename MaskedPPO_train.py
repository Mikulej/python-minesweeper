import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from enviornement import MineSweeper

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
#from stable_baselines3.common.monitor import Monitor
from time import sleep

#Train model
env = MineSweeper(renderMode="human",sizeX=16,sizeY=16,bombs=40)
#check_env(env)
observation, info = env.reset()
model = MaskablePPO("MlpPolicy", env,
                    learning_rate=0.01,
                    n_steps= 2048,
                    batch_size= 64,
                    n_epochs= 10,
                    gamma= 0.4)
model.learn(total_timesteps=10_000,use_masking=True)
print("Learning finished.")
# reward = evaluate_policy(model,env,n_eval_episodes=2)
# print("Mean reward is:",reward[0]," with error of: ",reward[1])
# if reward[0] >= 300:
#     print("Mean reward greater than 300, saving model...")
#     model.save("minesweepermodel")    
#     print("Model saved.")

#model.save("minesweepermodel")


for i in env.possible_actions:
   print("i: ",i," x: ", env.decode_action_x(i)," y: ", env.decode_action_y(i))
#print(env.possible_actions)
observation, info = env.reset()
#Test Model
i = 1
for _ in range(20):
    print(i,"score: ",env.score)
    i+=1
    print(observation)
    invalidActions = env.action_masks()
    print(invalidActions) 
    k = 0
    temp = []
    for ia in invalidActions:
       if ia == True:
        k+=1
        continue
       temp.append([env.decode_action_x(k),env.decode_action_y(k)])
       #print("x: ",env.decode_action_x(k),"y: ",env.decode_action_y(k))
       k+=1
    print(temp)
    action, states = model.predict(observation,action_masks=invalidActions,deterministic=False)
    print("x: ",env.decode_action_x(action), "y: ",env.decode_action_y(action))
    observation, reward, terminated,truncated, info = env.step(action)
    env.render(env.RENDER_MODE)
    
    
    #sleep(10000)
    if terminated or truncated:
      observation, info = env.reset()
      print(info)
env.close()

#https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
#https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/envs/invalid_actions_env.py