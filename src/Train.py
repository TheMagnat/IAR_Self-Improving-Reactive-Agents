import matplotlib
from tqdm import tqdm

import environment
from datetime import datetime
import Map

from torch.utils.tensorboard import SummaryWriter

from DQNAgent import DQNAgent, DQNAgent_article

import os

matplotlib.use("TkAgg")


writer = SummaryWriter("runs_DQN/runs"+datetime.now().strftime("%Y%m%d-%H%M%S"))



def classicTrain(agentName, env):

    playsCount = 300

    trainPerPlay = 20
    testPerPlay = 50

    freqSave = 30

    agent = DQNAgent(env.rotation)

    ###Simulation definition
    def simulation(env, agent, test=False):

        agent.test = test

        j = 0
        rsum = 0
        ob = env.reset()
        
        while True:
            
            action = agent.act(ob)
            new_ob, reward, done = env.step(action)
            
            j += 1

            if not test:
                agent.store(ob, action, new_ob, reward, done, j)
                agent.learn()
                    
            ob = new_ob

            #to save
            rsum += reward
            
            if done:
                break

        return rsum, env.foodEaten()
    ###Simulation definition


    csvFile = open(f'{agentName}.csv', 'w')
    csvFile.write('play,mean_rewards,mean_food_eaten\n')

    for i in tqdm(range(playsCount)):

        
        for t_i in range(trainPerPlay):
            rsum, foodEaten = simulation(env, agent, test=False)


        meanRsum, meanFoodEaten = 0, 0
        for t_i in range(testPerPlay):

            rsum, foodEaten = simulation(env, agent, test=True)

            meanRsum += rsum
            meanFoodEaten += foodEaten

        meanRsum /= testPerPlay
        meanFoodEaten /= testPerPlay

        csvFile.write(f'{i+1},{meanRsum},{meanFoodEaten}\n')
        
        if (i+1) % freqSave == 0 or (i+1) == playsCount:
            agent.save(f'DQN/save_{i+1}')

    csvFile.close()


def playThenBatchTrain(agentName, env, mode=0):

    playsCount = 300

    trainPerPlay = 20
    testPerPlay = 50

    freqSave = 30

    prior = False
    paperPrior = False

    if mode == 0:
        pass
    elif mode == 1:
        prior = True
    elif mode == 2:
        paperPrior = True
    elif mode == 3:
        pass


    agent = DQNAgent_article(env.rotation, N=100, batch_size=12, prior=prior, paperPrior=paperPrior)

    ###Simulation definition
    def simulation(env, agent, test=False):

        agent.test = test

        j = 0
        rsum = 0
        ob = env.reset()
        
        while True:
            
            action = agent.act(ob)
            new_ob, reward, done = env.step(action)
            
            j += 1

            if not test:
                agent.storeAndLearn(ob, action, new_ob, reward, done, j)
                    
            ob = new_ob

            rsum += reward
            
            if done:
                break

        return rsum, env.foodEaten()
    ###Simulation definition


    csvFile = open(f'{agentName}.csv', 'w')
    csvFile.write('play,mean_rewards,mean_food_eaten\n')

    for i in tqdm(range(playsCount)):

        
        for t_i in range(trainPerPlay):
            rsum, foodEaten = simulation(env, agent, test=False)


        meanRsum, meanFoodEaten = 0, 0
        for t_i in range(testPerPlay):

            rsum, foodEaten = simulation(env, agent, test=True)

            meanRsum += rsum
            meanFoodEaten += foodEaten

        meanRsum /= testPerPlay
        meanFoodEaten /= testPerPlay

        csvFile.write(f'{i+1},{meanRsum},{meanFoodEaten}\n')

        #Store the course
        agent.test = False
        agent.saveCourse()

        #Mode 3 is without buffer
        if mode != 3:
            agent.batchlLearn()

        
        if (i+1) % freqSave == 0 or (i+1) == playsCount:
            agent.save(f'DQN/save_{i+1}')


    csvFile.close()



if __name__ == '__main__':

    mode = 0

    Map1 = Map.Map1.copy()
    env = environment.Environment(foodNumber=15, env_map=Map1)
    
    if mode == 0:
        method = "DQN-r(random)"
    elif mode == 1:
        method = "DQN-r(prior)"
    elif mode == 2:
        method = "DQN-r(article)"
    elif mode == 3:
        method = "DQN"
    

    directory = f"runs_v2/{method}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    #classicTrain(f'runs/{method}/{method}_{datetime.now().strftime("%m_%d-%H_%M_%S")}', env)

    playThenBatchTrain(f'{directory}/{method}_{datetime.now().strftime("%m_%d-%H_%M_%S")}', env, mode)
