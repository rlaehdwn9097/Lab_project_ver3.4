import matplotlib.pyplot as plt
import config as cf
import dqn_learn as dl
from time import strftime, localtime, time
import os

class savingfunctions():

    def __init__(self):

        #self.save_epi_reward = []
        #self.save_epi_cache_hit_rate = []
        #self.save_epi_reward = []
        #self.save_epi_redundancy = []

        tm = localtime(time())
        self.Date = strftime('%Y-%m-%d_%H-%M-%S', tm)
        self.folderName = "LabResults/" + str(self.Date)
        os.mkdir(self.folderName)
        self.result_file = open(self.folderName +"/result.txt",'a')
        self.meta_file = open(self.folderName +"/metafile.txt",'a')

    
    ## save them to file if done
    def plot_reward_result(self, save_epi_reward):
        plt.plot(save_epi_reward)
        plt.savefig(self.folderName + '/rewards.png')
        plt.show()

    ## save them to file if done
    def plot_cache_hit_result(self, save_epi_cache_hit_rate):
        plt.plot(save_epi_cache_hit_rate)
        plt.savefig(self.folderName + '/cache_hit_rate.png')
        plt.show()

    def plot_redundancy_result(self, save_epi_redundancy):
        plt.plot(save_epi_redundancy)
        plt.savefig(self.folderName + '/redundancy.png')
        plt.show()

    def plot_denominator_result(self, denominator):
        plt.plot(denominator)
        plt.savefig(self.folderName + '/denominator.png')
        plt.show()

    def write_result_file(self, ep, time, NB_Round, step_cnt, action_cnt, cache_hit, cache_hit_rate, episode_reward, redundancy, avg_hop):
        self.result_file.write('Episode: ')
        self.result_file.write(str(ep+1))
        self.result_file.write('\t')
        self.result_file.write('Time: ')
        self.result_file.write(str(time))
        self.result_file.write('\t')
        self.result_file.write('NB_Round: ')
        self.result_file.write(str(NB_Round))
        self.result_file.write('\t')
        self.result_file.write('step_cnt: ')
        self.result_file.write(str(step_cnt))
        self.result_file.write('\t')
        self.result_file.write('action_cnt: ')
        self.result_file.write(str(action_cnt))
        self.result_file.write('\t')
        self.result_file.write('cache_hit: ')
        self.result_file.write(str(cache_hit))
        self.result_file.write('\t')
        self.result_file.write('cache_hit_rate : ')
        self.result_file.write(str(cache_hit_rate))
        self.result_file.write('\t')
        self.result_file.write('Reward: ')
        self.result_file.write(str(episode_reward))
        self.result_file.write('\t')
        self.result_file.write('redundancy : ')
        self.result_file.write(str(redundancy))
        self.result_file.write('\t')
        self.result_file.write('avg_hop : ')
        self.result_file.write(str(avg_hop))
        self.result_file.write('\n')

    def write_meta_file(self):
        self.meta_file.write("===Network simulator Meta Data===\n")
        self.meta_file.write('TOTAL_PRIOD = {}\n'.format(cf.TOTAL_PRIOD))
        self.meta_file.write('MAX_ROUNDS = {}\n'.format(cf.MAX_ROUNDS))
        self.meta_file.write('MAX_REQ_PER_ROUND = {}\n'.format(cf.MAX_REQ_PER_ROUND))
        self.meta_file.write('NB_NODES = {}\n'.format(cf.NB_NODES))
        self.meta_file.write('TX_RANGE = {}\n'.format(cf.TX_RANGE))
        self.meta_file.write('AREA_WIDTH = {}\n'.format(cf.AREA_WIDTH))
        self.meta_file.write('AREA_LENGTH = {}\n'.format(cf.AREA_LENGTH))
        self.meta_file.write('NUM_microBS = {}\n'.format(cf.NUM_microBS[0]*cf.NUM_microBS[1]))
        self.meta_file.write('NUM_BS = {}\n'.format(cf.NUM_BS[0]*cf.NUM_BS[1]))
        self.meta_file.write('microBS_SIZE = {}\n'.format(cf.microBS_SIZE))
        self.meta_file.write('BS_SIZE = {}\n'.format(cf.BS_SIZE))
        self.meta_file.write('CENTER_SIZE = {}\n'.format(cf.CENTER_SIZE))
        self.meta_file.write('DLthroughput = {}\n'.format(cf.DLthroughput))
        self.meta_file.write('ULthroughput = {}\n'.format(cf.ULthroughput))
        self.meta_file.write('DLpackets_per_second = {}\n'.format(cf.DLpackets_per_second))
        self.meta_file.write('ULpackets_per_second = {}\n'.format(cf.ULpackets_per_second))
        self.meta_file.write('LATENCY_INTERNET = {}\n'.format(cf.LATENCY_INTERNET))
        
        self.meta_file.write("\n")
        self.meta_file.write("===DQN structure Meta Data===\n")
        self.meta_file.write('H1 = {}\n'.format(cf.H1))
        self.meta_file.write('H2 = {}\n'.format(cf.H2))
        self.meta_file.write('H3 = {}\n'.format(cf.H3))
        self.meta_file.write('H4 = {}\n'.format(cf.H4))
        self.meta_file.write('q = {}\n'.format(cf.q))
        self.meta_file.write("\n")

        self.meta_file.write("===DQN Agent Meta Data===\n")
        self.meta_file.write('GAMMA = {}\n'.format(cf.GAMMA))
        self.meta_file.write('BATCH_SIZE = {}\n'.format(cf.BATCH_SIZE))
        self.meta_file.write('BUFFER_SIZE = {}\n'.format(cf.BUFFER_SIZE))
        self.meta_file.write('DQN_LEARNING_RATE = {}\n'.format(cf.DQN_LEARNING_RATE))
        self.meta_file.write('TAU = {}\n'.format(cf.TAU))
        self.meta_file.write('EPSILON = {}\n'.format(cf.EPSILON))
        self.meta_file.write('EPSILON_DECAY = {}\n'.format(cf.EPSILON_DECAY))
        self.meta_file.write('EPSILON_MIN = {}\n'.format(cf.EPSILON_MIN))
        self.meta_file.write('NB_ACTION = {}\n'.format(cf.NB_ACTION))
        self.meta_file.write("\n")

        self.meta_file.write("===Reward parameter Meta Data===\n")
        self.meta_file.write('a = {}\n'.format(cf.a))
        self.meta_file.write('b = {}\n'.format(cf.b))
        self.meta_file.write('c = {}\n'.format(cf.c))
        self.meta_file.write('d = {}\n'.format(cf.d))
        self.meta_file.write('e = {}\n'.format(cf.e))
     
