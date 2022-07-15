import matplotlib.pyplot as plt
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

    def write_meta_file(self):
        self.meta_file = open(self.folderName +"/metafile.txt",'a')
        self.meta_file.write()

    ## save them to file if done
    def plot_result(self, save_epi_reward):
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