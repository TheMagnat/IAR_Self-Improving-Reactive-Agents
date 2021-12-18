import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import enum
import random

import Map

colors = ['white', 'black', 'red', 'blue', 'orange', 'green','brown','yellow']
bounds = [0,1,2,3,4,5,6,7,8] #entre 0 et 1 exclu: white,...

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


class Actions(enum.Enum):
    B = 0
    G = 1
    H = 2
    D = 3


class Environment(object):
    CHARACTER = {
		0: ' ', #Vide
		1: 'O', #Obstacle
		2: 'E',	#Enemie
		3: '$',	#Food
		4: 'I',	#Agent
	}

    def __init__(self,foodNumber,env_map):
        self.foodNumber_init = foodNumber
        self.Map_init = np.array(env_map).copy()
        self.H = len(self.Map_init)
        self.V = len(self.Map_init[0])
        #ajout de bordure pour simplifier les fonctions de filtre des capteurs
        self.Map_init=np.concatenate((np.ones((len(self.Map_init),15)),self.Map_init,np.ones((len(self.Map_init),15))),axis=1)
        self.Map_init=np.concatenate((np.ones((15,len(self.Map_init[0]))),self.Map_init,np.ones((15,len(self.Map_init[0])))),axis=0)
        
        self.Y=np.array([[-1,-1,1,-1,-1],[-1,1,1,1,-1],[1,1,1,1,1],[-1,1,1,1,-1],[-1,-1,1,-1,-1]])
        self.O=np.ones((3,3))
        self.X=np.array([[-1,1,-1],[1,1,1],[-1,1,-1]])
        self.o=np.array([[1]])
        self.food_detector=[[[10,0],[8,-2],[6,-4],[4,-6],[2,-8],
                             [0,-10],[-2,-8],[-4,-6],[-6,-4],[-8,-2],
                             [-10,0],[-8,2],[-6,4],[-4,6],[-2,8],
                             [0,10],[2,8],[4,6],[6,4],[8,2]],
                            [[6,0],[4,-2],[2,-4],
                             [0,-6],[-2,-4],[-4,-2],
                             [-6,0],[-4,2],[-2,4],
                             [0,6],[2,4],[4,2],
                             [4,0],[2,-2],[0,-4],[-2,-2],[-4,0],[-2,2],[0,4],[2,2]],
                            [[2,0],[1,-1],[0,-2],[-1,-1],[-2,0],[-1,1],[0,2],[1,1],
                             [1,0],[0,-1],[-1,0],[0,1]],
                            []]
        self.enemie_detector=[[],
                              [[6,0],[4,-2],[2,-4],
                               [0,-6],[-2,-4],[-4,-2],
                               [-6,0],[-4,2],[-2,4],
                               [0,6],[2,4],[4,2],
                               [4,0],[2,-2],[0,-4],[-2,-2],[-4,0],[-2,2],[0,4],[2,2]],
                              [[2,0],[1,-1],[0,-2],[-1,-1],[-2,0],[-1,1],[0,2],[1,1],
                               [1,0],[0,-1],[-1,0],[0,1]],
                              []]
        self.obstacle_detector=[[],
                                [],
                                [],
                                [[4,0],[3,-1],[2,-2],[1,-3],
                                 [0,-4],[-1,-3],[-2,-2],[-3,-1],
                                 [-4,0],[-3,1],[-2,2],[-1,3],
                                 [0,4],[1,3],[2,2],[3,1],
                                 [3,0],[2,-1],[1,-2],
                                 [0,-3],[-1,-2],[-2,-1],
                                 [-3,0],[-2,1],[-1,2],
                                 [0,3],[1,2],[2,1],
                                 [2,0],[1,-1],[0,-2],[-1,-1],[-2,0],[-1,1],[0,2],[1,1],
                                 [1,0],[0,-1],[-1,0],[0,1]]]
        
        self.rotation=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,3,4,
                       23,24,25,26,27,28,29,30,31,20,21,22,
                       34,35,36,37,38,39,32,33,
                       42,43,44,45,46,47,40,41,
                       49,50,51,48,
                       55,56,57,58,59,60,61,62,63,52,53,54,
                       66,67,68,69,70,71,64,65,
                       74,75,76,77,78,79,72,73,
                       81,82,83,80,
                       88,89,90,91,92,93,94,95,96,97,98,99,84,85,86,87,
                       103,104,105,106,107,108,109,110,111,100,101,102,
                       114,115,116,117,118,119,112,113,
                       121,122,123,120,
                       124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,
                       141,142,143,140,
                       144]
        
        self.energy_max=40+15*self.foodNumber_init-self.foodNumber_init
        self.previous_action=np.zeros(len(Actions))
        self.colision_obstacle=0
        
        
    def reset(self):
        self.foodNumber=self.foodNumber_init
        self.Map=self.Map_init.copy()
        nb_cases_vide=(self.Map==0).sum()
        cases_vide=np.arange(nb_cases_vide)
        np.random.shuffle(cases_vide)
        cases_nourriture=cases_vide[:self.foodNumber]
        ind=0
        self.pos_agent=None
        self.pos_enemies=[]
        for ligne in range(len(self.Map)):
            for colonne in range(len(self.Map[0])):
                if self.Map[ligne,colonne]==0:
                    if ind in cases_nourriture:
                        self.Map[ligne,colonne]=3
                    ind+=1
                if self.Map[ligne,colonne]==2:
                    self.pos_enemies.append([ligne,colonne])
                if self.Map[ligne,colonne]==4:
                    self.pos_agent=[ligne,colonne]

        self.enemies_on_food=[0]*len(self.pos_enemies) #pour que la nourriture réaparraisse si un énemie parche dessus
        self.colision_enemie=0
        self.energy=40
        
        obs,obs_affichage=self.observation()
        
        self.previous_action=np.zeros(len(Actions))
        self.colision_obstacle=0
        
        return obs
        
    def observation(self):
        obs = []
        obs_affichage = np.ones((25,55))*7
        
        obs_affichage[12,33] = 4
        for type_capteur, pos_capteurs in enumerate(self.food_detector):
            if type_capteur == 0:
                longueur_filtre = 5
                filtre = self.Y

            elif type_capteur == 1:
                longueur_filtre = 3
                filtre = self.O

            elif type_capteur == 2:
                longueur_filtre = 3
                filtre = self.X

            elif type_capteur == 3:
                longueur_filtre = 1
                filtre = self.o

            for pos_capteur in pos_capteurs:
                valeur_filtre=filtre*3 == self.Map[int(self.pos_agent[0]+pos_capteur[0]-(longueur_filtre-1)/2):int(self.pos_agent[0]+pos_capteur[0]+(longueur_filtre-1)/2+1),int(self.pos_agent[1]+pos_capteur[1]-(longueur_filtre-1)/2):int(self.pos_agent[1]+pos_capteur[1]+(longueur_filtre-1)/2+1)]
                obs.append(int(valeur_filtre.sum()>=1))
                if valeur_filtre.sum()>=1:
                    obs_affichage[12+pos_capteur[0],12+pos_capteur[1]] = 3
        

        obs_affichage[12,12] = 4

        for type_capteur,pos_capteurs in enumerate(self.enemie_detector):
            if type_capteur==0:
                longueur_filtre=5
                filtre=self.Y
            elif type_capteur==1:
                longueur_filtre=3
                filtre=self.O
            elif type_capteur==2:
                longueur_filtre=3
                filtre=self.X
            elif type_capteur==3:
                longueur_filtre=1
                filtre=self.o
            for pos_capteur in pos_capteurs:
                valeur_filtre=filtre*2==self.Map[int(self.pos_agent[0]+pos_capteur[0]-(longueur_filtre-1)/2):int(self.pos_agent[0]+pos_capteur[0]+(longueur_filtre-1)/2+1),int(self.pos_agent[1]+pos_capteur[1]-(longueur_filtre-1)/2):int(self.pos_agent[1]+pos_capteur[1]+(longueur_filtre-1)/2+1)]
                obs.append(int(valeur_filtre.sum()>=1))
                if valeur_filtre.sum()>=1:
                    obs_affichage[12+pos_capteur[0],33+pos_capteur[1]]=2
                # if valeur_filtre.sum()>=1:
                #     print(valeur_filtre)
                #     print(self.Map[int(self.pos_agent[0]+pos_capteur[0]-(longueur_filtre-1)/2):int(self.pos_agent[0]+pos_capteur[0]+(longueur_filtre-1)/2+1),int(self.pos_agent[1]+pos_capteur[1]-(longueur_filtre-1)/2):int(self.pos_agent[1]+pos_capteur[1]+(longueur_filtre-1)/2+1)])
                #     print(filtre*2)
                #     time.sleep(200)
                
        obs_affichage[12,48]=4
        for type_capteur,pos_capteurs in enumerate(self.obstacle_detector):
            if type_capteur==0:
                longueur_filtre=5
                filtre=self.Y
            elif type_capteur==1:
                longueur_filtre=3
                filtre=self.O
            elif type_capteur==2:
                longueur_filtre=3
                filtre=self.X
            elif type_capteur==3:
                longueur_filtre=1
                filtre=self.o
            for pos_capteur in pos_capteurs:
                valeur_filtre=filtre*1==self.Map[int(self.pos_agent[0]+pos_capteur[0]-(longueur_filtre-1)/2):int(self.pos_agent[0]+pos_capteur[0]+(longueur_filtre-1)/2+1),int(self.pos_agent[1]+pos_capteur[1]-(longueur_filtre-1)/2):int(self.pos_agent[1]+pos_capteur[1]+(longueur_filtre-1)/2+1)]
                obs.append(int(valeur_filtre.sum()>=1))
                if valeur_filtre.sum()>=1:
                    obs_affichage[12+pos_capteur[0],48+pos_capteur[1]]=1
        
        
        for i in range(16):
            if i<=round(16/self.energy_max*self.energy):
                obs.append(1)
            else:
                obs.append(0)
        for i in range(4):
            obs.append(int(self.previous_action[i]))
            
        obs.append(self.colision_obstacle)
        
        return obs,obs_affichage
        
        
    def render(self,show=True):
        obs,obs_affichage=self.observation()
        OBS=np.ones(self.V)*7
        OBS[:16]=6
        OBS[:16]-=obs[-21:-5]
        
        OBS[18:22]=6
        OBS[18:22]-=obs[-5:-1]
        
        OBS[-1]=6
        OBS[-1]-=obs[-1]
        
        # Affichage=np.concatenate((self.Map[15:-15,15:-15],OBS.reshape(1,-1)),axis=0)
        Affichage=np.concatenate((obs_affichage,self.Map[15:-15],np.concatenate((np.ones((1,15)),OBS.reshape(1,-1),np.ones((1,15))),axis=1)),axis=0)
        
        # plt.figure()
        # plt.imshow(obs_affichage,cmap=cmap, norm=norm)
        if show:
            plt.figure()
            plt.imshow(Affichage,cmap=cmap, norm=norm)
            plt.show(block=False)
            # plt.pause(0.5)
            # plt.close()
        return Affichage
        
    def step(self, action):
        self.previous_action=np.zeros(len(Actions))
        self.previous_action[action]=1
        self.colision_obstacle=0
        done=False
        reward=self.move_agent(action)
        self.move_enemies()
        if self.foodNumber==0 or self.colision_enemie==1 or self.energy==0:
            done=True
        if self.colision_enemie==1 or self.energy==0:
            reward=-1
        obs,obs_affichage=self.observation()
        return obs,reward,done

    
    def move_agent(self,action):
        reward=0
        action = Actions(action)
        if action==Actions.H:
            h=-1
            v=0
        elif action==Actions.B:
            h=1
            v=0
        elif action==Actions.G:
            h=0
            v=-1
        elif action==Actions.D:
            h=0
            v=1
        else:
            print('Mauvaise action :',action)
        next_case=[self.pos_agent[0]+h,self.pos_agent[1]+v]
        if self.Map[next_case[0],next_case[1]]==1:
            self.colision_obstacle=1
            
        if self.Map[next_case[0],next_case[1]]==2:
            self.colision_enemie=1
        if self.Map[next_case[0],next_case[1]]!=1 and self.Map[next_case[0]][next_case[1]]!=2:
            if self.Map[next_case[0],next_case[1]]==3:
                self.foodNumber-=1
                self.energy+=15
                reward=0.4
                
            self.Map[self.pos_agent[0],self.pos_agent[1]]=0
            self.Map[next_case[0],next_case[1]]=4
            self.pos_agent=next_case
        self.energy-=1
        return reward
        
    def move_enemies(self):
        for enemie,pos_enemie in enumerate(self.pos_enemies):
            if random.random()<0.2:
                pass
            else:
                P=[]
                
                dist=np.linalg.norm(np.array(self.pos_agent)-np.array(pos_enemie))
                if dist==0: #s'il y a eu collision avec l'agent
                    pass 
                else:
                    dir_enemie_agent=np.array(self.pos_agent)-np.array(pos_enemie)
                    if dir_enemie_agent[0]>=0:
                        if dir_enemie_agent[1]>=0:
                            angle_enemie_agent=-360/(2*math.pi)*math.acos(dir_enemie_agent[0]/dist)
                        else:
                            angle_enemie_agent=360/(2*math.pi)*math.acos(dir_enemie_agent[0]/dist)
                    else:
                        if dir_enemie_agent[1]>=0:
                            angle_enemie_agent=-360/(2*math.pi)*(math.pi-math.acos(np.abs(dir_enemie_agent[0])/dist))
                        else:
                            angle_enemie_agent=360/(2*math.pi)*(math.pi-math.acos(np.abs(dir_enemie_agent[0])/dist))
                    
                    # print(angle_enemie_agent)
                    # print(dist)
                    for i in range(4):
                        if i==0: #si on va vers le bas
                            h=1
                            v=0
                        elif i==1: #si on va vers la gauche
                            h=0
                            v=-1
                        elif i==2:
                            h=-1
                            v=0
                        elif i==3:
                            h=0
                            v=1
                        next_case=[pos_enemie[0]+h,pos_enemie[1]+v]
                        if self.Map[next_case[0],next_case[1]]==1:
                            P.append(0)
                        else:
                            angle_A=90*i
                            angle=angle_A-angle_enemie_agent
               
                            #pas demandé mais permet de donner du sens à W_angle
                            #if faut que angle soit compris entre -180 et 180 degrés
                            if angle>180: 
                                angle=angle-360
                            elif angle<-180:
                                angle=angle+360
                                
                            W_angle=(180-np.abs(angle))/180
                            
                            if dist<=4:
                                T_dist=15-dist
                            elif dist<=15:
                                T_dist=9-dist/2
                            else:
                                T_dist=1
                            P.append(math.exp(0.33*W_angle*T_dist))
                    P=np.array(P)
                    P=P/P.sum()
                    # print(P)
                    action=np.random.choice(np.arange(4),p=P)
                    if action==0: #si on va vers le bas
                        h=1
                        v=0
                    elif action==1: #si on va vers la gauche
                        h=0
                        v=-1
                    elif action==2:
                        h=-1
                        v=0
                    elif action==3:
                        h=0
                        v=1
                    next_case=[pos_enemie[0]+h,pos_enemie[1]+v]
                    
                    if self.enemies_on_food[enemie]:
                        self.Map[pos_enemie[0],pos_enemie[1]]=3
                        self.enemies_on_food[enemie]=0
                    else:
                        self.Map[pos_enemie[0],pos_enemie[1]]=0
                    
                    if self.Map[next_case[0]][next_case[1]]==3:
                        self.enemies_on_food[enemie]=1
                    
                    if self.Map[next_case[0]][next_case[1]]==4:
                        self.colision_enemie=1
                        
                    self.Map[next_case[0],next_case[1]]=2
                    self.pos_enemies[enemie]=[next_case[0],next_case[1]]
        

#########################################


if __name__ == '__main__':
    Map1=Map.Map1.copy()
    env = Environment(foodNumber=15,env_map=Map1)

    obs=env.reset()
    env.render()
    while True:
        caractere=input()
        if caractere=='r':
            obs=env.reset()
            env.render()
            action=None

        elif caractere=='s':
            action=0
        elif caractere=='q':
            action=1
        elif caractere=='z':
            action=2
        elif caractere=='d':
            action=3
        elif caractere=='e':
            break
        else:
            print('Mauvais caractère :',caractere)
        
        try:
            obs,reward,done=env.step(action)
            # print(obs)
            print(reward)
            env.render()
            if done:
                obs=env.reset()
                env.render()
        except:
            pass
        

# while True:
#     with keyboard.Events() as events:
#         event = events.get()
#         try:
#             print(event.key)
#         except:
#             pass