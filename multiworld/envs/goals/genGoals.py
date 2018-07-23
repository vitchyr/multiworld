

import pickle

import numpy as np

save_dir = '/home/russellm/multiworld/multiworld/envs/goals/'

def gen_goals(fileName):

    goals = []
    for count in range(20):

        i = np.random.uniform(-.1, .1)
        j = np.random.uniform(0.6, 0.8)

        goals.append([i,j])


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(goals, fobj)
    fobj.close()

def read_goals(fileName):

    fobj = open(save_dir+fileName+'.pkl', 'rb')
    goals = pickle.load(fobj)
    return goals



def visualize(fileName):

    goals = np.array(read_goals(fileName))
    from matplotlib import pyplot as plt

    xs = goals[:,0]
    ys = goals[:,1]

    
    
    plt.scatter(xs, ys)

    plt.savefig(fileName+'.png')


#gen_goals('sawyer_pick_goals_file1')

#visualize('sawyer_pick_goals_file1')
goals = read_goals('sawyer_pick_goals_file1')
import ipdb
ipdb.set_trace()







