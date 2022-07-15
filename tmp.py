import random
import config as cf
import gym
import scenario as sc
from time import strftime, localtime, time
import os 


list1 = [10, 12, 13, 0, 0]

tmp = min(list1)
index = list1.index(tmp)

print(index)
print(list1[index])


"""
tm = localtime(time())
Date = strftime('%Y-%m-%d_%H-%M-%S', tm)
folderName = "LabResults/" + str(Date)
print(folderName)
os.mkdir(folderName)
"""


"""
env_name = 'CartPole-v1'
env = gym.make(env_name)
print(env.env.__dict__)


print(len(sc.emBB))
print(sc.emBB[random.randrange(0,len(sc.emBB))].__dict__)
"""
