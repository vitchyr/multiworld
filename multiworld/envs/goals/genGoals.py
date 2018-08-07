

import pickle

import numpy as np

save_dir = '/home/russellm/multiworld/multiworld/envs/goals/'


def read_goals(fileName):

    fobj = open(save_dir+fileName+'.pkl', 'rb')
    goals = pickle.load(fobj)

    import ipdb
    ipdb.set_trace()
    return goals

def modify(oldName, newName):

    tasks = read_goals(oldName)


    new_tasks = []

    
    for old_task in tasks:

        new_task={}
       
        new_task['obj_init_pos'] = np.concatenate([old_task['obj'], [0.02]])
        new_task['goal'] = np.concatenate([old_task['goal'], [0.02]]) 
        new_task['height'] = 0.1

        new_tasks.append(new_task)

    

    fobj = open(save_dir+newName+'.pkl', 'wb')
    pickle.dump(new_tasks, fobj)
    fobj.close()

#modify('pickPlace_60X30_old', 'pickPlace_60X30')
read_goals('pickPlace_60X30')


def gen_pickPlaceGoals(fileName):

    goals = []
    for count in range(20):

        task={}

        xs = np.random.uniform(-.3, .3, size=2)
        ys = np.random.uniform(0.5, 0.8, size = 2)

        task['obj'] = [xs[0], ys[0]]
        task['goal'] = [xs[1], ys[1]]

        goals.append(task)


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(goals, fobj)
    fobj.close()

def visualize_pickPlace(fileName, xRange = [-.3, .3], yRange=[0.5, 0.8]):

    tasks = np.array(read_goals(fileName))
    from matplotlib import pyplot as plt

    for i in range(len(tasks)):
        task = tasks[i]

        color = np.random.uniform(0,1, size=3)
        plt.annotate(xy = task['obj'], s= 'O'+str(i), color=color)
        plt.annotate(xy = task['goal'], s='G'+str(i), color=color)


    plt.xlim(xRange[0], xRange[1])
    plt.ylim(yRange[0], yRange[1])


    plt.savefig(fileName+'.png')


#gen_pickPlaceGoals('pickPlace_60X30')


#visualize_pickPlace('pickPlace_60X30')

#read_goals('pickPlace_60X30')




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







def visualize(fileName):

    goals = np.array(read_goals(fileName))[:20]
    from matplotlib import pyplot as plt




    xs = goals[:,0]
    ys = goals[:,1]

    
    
    plt.scatter(xs, ys)

    plt.savefig(fileName+'.png')






#read_goals('sawyer_pick_goals_60X35_train')



## goals = read_goals('sawyer_pick_goals_file1')
# import ipdb
# ipdb.set_trace()







