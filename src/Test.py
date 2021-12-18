import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import environment
from datetime import datetime
import Map


from torch.utils.tensorboard import SummaryWriter

from DQNAgent import DQNAgent

colors = ['white', 'black', 'red', 'blue', 'orange', 'green','brown','yellow']
bounds = [0,1,2,3,4,5,6,7,8] #entre 0 et 1 exclu: white,...

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

writer = SummaryWriter("runs/runs"+datetime.now().strftime("%Y%m%d-%H%M%S"))
            
            

if __name__ == '__main__':
    Map1=Map.Map1.copy()
    env = environment.Environment(foodNumber=15,env_map=Map1)
    episode_count = 10000000

    agent = DQNAgent(env.rotation)
    agent.load('DQN/save_4900')
    agent.test = True

    rsum = 0
    mean = 0
    itest = 0
    reward = 0
    done = False
    
    for i in range(episode_count):
        
        rsum = 0
        ob = env.reset()
        
        init_affichage=0
        j = 0
        
        while True:
            j+=1
            action= agent.act(ob)
            new_ob, reward, done = env.step(action)
            ob=new_ob
            Affichage=env.render(show=False)
            
            if init_affichage==0:
                A=plt.imshow(Affichage,cmap=cmap, norm=norm)
                init_affichage=1
            else:
                A.set_data(Affichage)
            plt.pause(0.1)
            rsum += reward
            if done:
                init_affichage=0
                plt.show(block=False)
                plt.pause(0.1)
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                writer.add_scalar("reward", rsum, i)
                mean += rsum
                rsum = 0
                break


