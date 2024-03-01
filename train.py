import time
def evaluate_mask(model, env, timesteps):
    scores = 0
    numOfGames = 0
    numOfWins = 0
    observation, info = env.reset()
    for _ in range(timesteps):
        
        invalidActions = env.action_masks()
        action, states = model.predict(observation,action_masks=invalidActions,deterministic=False)
        observation, reward, terminated,truncated, info = env.step(action)
        env.render(env.RENDER_MODE)
        #time.sleep(1)
        if terminated or truncated:
            scores += env.score
            numOfGames += 1
            if truncated:
                numOfWins += 1
            observation, info = env.reset()
            print(info)
    return [scores / numOfGames, 100 * numOfWins / numOfGames]

def evaluate(model, env, timesteps):
    scores = 0
    numOfGames = 0
    numOfWins = 0
    observation, info = env.reset()
    for _ in range(timesteps): 
        action, states = model.predict(observation,deterministic=False)
        observation, reward, terminated,truncated, info = env.step(action)
        env.render(env.RENDER_MODE)
        #time.sleep(1)
        if terminated or truncated:
            scores += env.score
            numOfGames += 1
            if truncated:
                numOfWins += 1
            observation, info = env.reset()
            print(info)
    return [scores / numOfGames, 100 * numOfWins / numOfGames]