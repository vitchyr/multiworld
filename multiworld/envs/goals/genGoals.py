

import pickle

import numpy as np

save_dir = '/home/russellm/multiworld/multiworld/envs/goals/'

def gen_pickPlaceGoals(fileName):

    goals = []
    for count in range(20):

        # i = np.random.uniform(-.1, .1)
        # j = np.random.uniform(0.6, 0.8)

        i = np.random.uniform(-.3, .3)
        j = np.random.uniform(0.5, 0.8)

        goals.append([i,j])


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(goals, fobj)
    fobj.close()


def gen_pointMassGoals(fileName):

    goals = []
    for count in range(20):


        theta = np.random.uniform(0, 2*np.pi)
        # i = np.random.uniform(-.1, .1)
        # j = np.random.uniform(0.6, 0.8)

        i = 0.2* np.cos(theta)
        j = 0.2*np.sin(theta)

        goals.append([i,j])


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(goals, fobj)
    fobj.close()


def gen_objPos_goalPos(fileName):

    goals = []
    for count in range(20):

        # i = np.random.uniform(-.1, .1)
        # j = np.random.uniform(0.6, 0.8)

        config = {}
        i1 = np.random.uniform(-.3, .3)
        j1 = np.random.uniform(0.5, 0.65)

        i2 = np.random.uniform(-.3, .3)
        j2 = np.random.uniform(.7, .85)

        config['objPos'] = [i1, j1]
        config['goalPos'] = [i2, j2]

        goals.append(config)


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(goals, fobj)
    fobj.close()


#gen_objPos_goalPos('obj_behind_goal')



def read_goals(fileName):

    fobj = open(save_dir+fileName+'.pkl', 'rb')
    goals = pickle.load(fobj)

    import ipdb
    ipdb.set_trace()
    return goals





def visualize(fileName):

    goals = np.array(read_goals(fileName))[:20]
    from matplotlib import pyplot as plt

    xs = goals[:,0]
    ys = goals[:,1]

    
    
    plt.scatter(xs, ys)

    plt.savefig(fileName+'.png')






read_goals('sawyer_pick_goals_60X35_train')

# gen_pointMassGoals('pointMassVal')


# visualize('pointMassVal')

## goals = read_goals('sawyer_pick_goals_file1')
# import ipdb
# ipdb.set_trace()







