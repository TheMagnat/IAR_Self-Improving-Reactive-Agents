import matplotlib
matplotlib.use("TkAgg")
import environment
from datetime import datetime
import Map



from torch.utils.tensorboard import SummaryWriter

from DQNAgent import DQNAgent

writer = SummaryWriter("runs_DQN/runs"+datetime.now().strftime("%Y%m%d-%H%M%S"))



if __name__ == '__main__':
    Map1=Map.Map1.copy()
    env = environment.Environment(foodNumber=15,env_map=Map1)

    freqTest = 100
    freqSave = 100
    nbTest = 10
    episode_count= 5000

    agent = DQNAgent(env.rotation)

    rsum = 0
    mean = 0
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        
        rsum = 0
        ob = env.reset()

        
        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            writer.add_scalar("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save('DQN' + "/save_" + str(i))

        j = 0
        
        while True:
            
            action= agent.act(ob)
            # print(action)
            new_ob, reward, done = env.step(action)
            
            j+=1

            agent.store(ob, action, new_ob, reward, done,j)
            ob=new_ob
            rsum += reward
            
            agent.learn()
            
            if done:
                
                # print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ","    ")
                writer.add_scalar("reward", rsum, i)
                mean += rsum
                rsum = 0

                break


