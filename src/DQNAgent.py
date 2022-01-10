
from core import NN
from memory import Memory, MemoryCourse
import torch
import numpy as np
import random


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    Device = torch.device('cuda')
else:
    Device = torch.device('cpu')


class DQNAgent(object):
    """The world's simplest agent!"""

    def __init__(self, rotation, N=10000, batch_size=32, prior=False, paperPrior=False):
        self.test = False
        self.net = NN(145, 1, layers=[512], finalActivation=None, activation=torch.nn.ReLU(),dropout=0.0).to(Device)
        self.target_net = NN(145, 1, layers=[512], finalActivation=None, activation=torch.nn.ReLU(),dropout=0.0).to(Device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.epsilon = 1
        self.gamma = 0.9
        self.lr = 1e-3
        self.N = N
        self.C = 1000
        self.batch_size = batch_size
        self.prior = prior
        self.paperPrior = paperPrior
        self.buffer = Memory(self.N, prior=self.prior, paperPrior=self.paperPrior)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.nb_iter = 0

        self.rotation = rotation
        

    def act(self, obs):
        obs=np.array(obs)
        if self.test:
            Q=np.zeros(4)
            for a in range(4):
                Q[a]=self.net(torch.tensor([obs],dtype=torch.float32,device=Device)).detach().cpu().numpy()[0]
                obs=obs[self.rotation]
            return np.argmax(Q)
        else:
            self.epsilon=max(0.1,self.epsilon-1/150000)
            self.nb_iter+=1
            if random.random()<self.epsilon:
                return random.randint(0,3)
            else:
                Q=np.zeros(4)
                for a in range(4):
                    Q[a]=self.net(torch.tensor([obs],dtype=torch.float32,device=Device)).detach().cpu().numpy()[0]
                    obs=obs[self.rotation]
            return np.argmax(Q)
            
    def save(self,outputDir):
        # with open(outputDir,'wb') as f:
        #     pickle.dump(self.net,f)
        torch.save(self.net.state_dict(),outputDir)

    def load(self,inputDir):
        # with open(inputDir,'rb') as f:
        #     self.net=pickle.load(f)
        self.net.load_state_dict(torch.load(inputDir, map_location=Device))

    def learn(self):
        if self.test:
            return
        else:
            self.optimizer.zero_grad()
            idx, w, mini_batch=self.buffer.sample(self.batch_size)
            ob, action, reward, new_ob, done=map(list,zip(*mini_batch))
            ob_v=torch.tensor(np.array(ob).reshape(-1,145),dtype=torch.float32,device=Device)
            action_v=torch.tensor(np.array(action).reshape(-1),dtype=torch.float32,device=Device)
            new_ob_v=torch.tensor(np.array(new_ob).reshape(-1,145),dtype=torch.float32,device=Device)
            reward_v=torch.tensor(np.array(reward).reshape(-1),dtype=torch.float32,device=Device)
            done_v=torch.tensor(np.array(done).reshape(-1),dtype=torch.float32,device=Device)
            
            Q_next=torch.zeros((self.batch_size,4)).to(Device)
            for a in range(4):
                Q_next[:,a]=self.target_net(new_ob_v).view(-1)
                new_ob_v=new_ob_v[[[i]*145 for i in range(self.batch_size)],[self.rotation for _ in range(self.batch_size)]]
            y=reward_v+(1-done_v)*self.gamma*torch.max(Q_next,dim=1)[0]
            #Le y.detach() est nécessaire pour ne pas calculer le gradient de y innutilement et perdre du temps
            # loss=torch.nn.SmoothL1Loss()(y.detach(),self.net(ob_v)[range(len(action_v)),np.array(action_v.detach().cpu().numpy())])
            Q=torch.zeros((self.batch_size,4)).to(Device)
            for a in range(4):
                Q[:,a]=self.net(ob_v).view(-1)
                ob_v=ob_v[[[i]*145 for i in range(self.batch_size)],[self.rotation for _ in range(self.batch_size)]]
            loss=torch.nn.MSELoss()(y.detach(),Q[range(self.batch_size),np.array(action_v.detach().cpu().numpy())])
            
            loss.backward()
            # print(loss)
            self.optimizer.step()
            if self.nb_iter%self.C==0:
                self.target_net.load_state_dict(self.net.state_dict())
        

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.buffer.store(tr)
            

class DQNAgent_article(object):
    """The world's simplest agent!"""

    def __init__(self, rotation, N=100, batch_size=12, prior=False, paperPrior=False):
        self.test = False
        self.net = NN(145, 1, layers=[512], finalActivation=None, activation=torch.nn.ReLU(),dropout=0.0).to(Device)
        self.target_net = NN(145, 1, layers=[512], finalActivation=None, activation=torch.nn.ReLU(),dropout=0.0).to(Device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.epsilon = 1
        self.gamma = 0.9
        self.lr = 1e-3
        self.N = N
        self.C = 1000
        self.batch_size = batch_size
        self.prior = prior
        self.paperPrior = paperPrior
        self.buffer = Memory(self.N, prior=self.prior, paperPrior=self.paperPrior)

        self.currentCourseBuffer = MemoryCourse()

        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.nb_iter = 0

        self.rotation = rotation
        

    def act(self, obs):
        obs=np.array(obs)
        if self.test:
            Q=np.zeros(4)
            for a in range(4):
                Q[a]=self.net(torch.tensor([obs],dtype=torch.float32,device=Device)).detach().cpu().numpy()[0]
                obs=obs[self.rotation]
            return np.argmax(Q)
        else:
            self.epsilon=max(0.1,self.epsilon-1/150000)
            self.nb_iter+=1
            if random.random()<self.epsilon:
                return random.randint(0,3)
            else:
                Q=np.zeros(4)
                for a in range(4):
                    Q[a]=self.net(torch.tensor([obs],dtype=torch.float32,device=Device)).detach().cpu().numpy()[0]
                    obs=obs[self.rotation]
            return np.argmax(Q)
            
    def save(self,outputDir):
        # with open(outputDir,'wb') as f:
        #     pickle.dump(self.net,f)
        torch.save(self.net.state_dict(),outputDir)

    def load(self,inputDir):
        # with open(inputDir,'rb') as f:
        #     self.net=pickle.load(f)
        self.net.load_state_dict(torch.load(inputDir, map_location=Device))


    def batchlLearn(self):
        if self.test:
            return

        idx, w, courses = self.buffer.sample(self.batch_size)

        for course in courses:
            self.learn(course.getAll())

    def learn(self, batch):
        if self.test:
            return
        else:

            batch_size = len(batch)

            self.optimizer.zero_grad()

            #idx, w, mini_batch=self.buffer.sample(self.batch_size)

            ob, action, reward, new_ob, done = map(list,zip(*batch))

            ob_v=torch.tensor(np.array(ob).reshape(-1,145),dtype=torch.float32,device=Device)
            action_v=torch.tensor(np.array(action).reshape(-1),dtype=torch.float32,device=Device)
            new_ob_v=torch.tensor(np.array(new_ob).reshape(-1,145),dtype=torch.float32,device=Device)
            reward_v=torch.tensor(np.array(reward).reshape(-1),dtype=torch.float32,device=Device)
            done_v=torch.tensor(np.array(done).reshape(-1),dtype=torch.float32,device=Device)

            Q_next=torch.zeros((batch_size,4)).to(Device)
            for a in range(4):
                Q_next[:,a]=self.target_net(new_ob_v).view(-1)
                new_ob_v=new_ob_v[[[i]*145 for i in range(batch_size)],[self.rotation for _ in range(batch_size)]]
            y=reward_v+(1-done_v)*self.gamma*torch.max(Q_next,dim=1)[0]
            #Le y.detach() est nécessaire pour ne pas calculer le gradient de y innutilement et perdre du temps
            # loss=torch.nn.SmoothL1Loss()(y.detach(),self.net(ob_v)[range(len(action_v)),np.array(action_v.detach().cpu().numpy())])
            Q=torch.zeros((batch_size,4)).to(Device)
            for a in range(4):
                Q[:,a]=self.net(ob_v).view(-1)
                ob_v=ob_v[[[i]*145 for i in range(batch_size)],[self.rotation for _ in range(batch_size)]]
            loss=torch.nn.MSELoss()(y.detach(),Q[range(batch_size),np.array(action_v.detach().cpu().numpy())])
            
            loss.backward()
            # print(loss)
            self.optimizer.step()
            if self.nb_iter%self.C==0:
                self.target_net.load_state_dict(self.net.state_dict())
        

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.buffer.store(tr)

    def storeAndLearn(self, ob, action, new_ob, reward, done, it):
        if not self.test:
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition = tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.currentCourseBuffer.store(tr)


            last = self.currentCourseBuffer.getLast()

            self.learn([last])

    def saveCourse(self):
        self.buffer.store(self.currentCourseBuffer)
        self.currentCourseBuffer = MemoryCourse()
            