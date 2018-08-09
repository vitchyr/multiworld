

import pickle

import numpy as np

save_dir = '/home/russellm/multiworld/multiworld/envs/goals/'


def read_goals(fileName):

    fobj = open(save_dir+fileName+'.pkl', 'rb')
    goals = pickle.load(fobj)

   
    return goals



def gen_doorGoals(fileName):

    tasks = []
    for count in range(20):

        task={}

        task['door_init_pos'] = np.random.uniform([-.3,0.8, 0.2] ,[.3,1, 0.4])
        task['goalAngle'] = np.random.uniform([0], [1.5708])

        tasks.append(task)

    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(tasks, fobj)
    fobj.close()




def visualize_doorPos(fileName, xRange = [-.3, .3], yRange=[0.8, 1.0]):

    tasks = np.array(read_goals(fileName))
    from matplotlib import pyplot as plt

    for i in range(len(tasks)):
        task = tasks[i]

        color = np.random.uniform(0,1, size=3)
        plt.annotate(xy = task['door_init_pos'][:2], s= 'O'+str(i), color=color)
        #plt.annotate(xy = task['goal'], s='G'+str(i), color=color)


    plt.xlim(xRange[0], xRange[1])
    plt.ylim(yRange[0], yRange[1])


    plt.savefig(fileName+'.png')




#gen_doorGoals('doorOpening_60X20X20')


#visualize_doorPos('doorOpening_60X20X20')








def gen_pickPlaceGoals(fileName):

    tasks = []
    for count in range(20):

        task={}

        xs = np.random.uniform(-.1, .1, size=2)
        ys = np.random.uniform(0.6, 0.8, size = 2)

        task['obj_init_pos'] = np.array([xs[0], ys[0], 0.02])
        task['goal'] = np.array([xs[1], ys[1], 0.02])
        task['height'] = 0.06

        tasks.append(task)


    fobj = open(save_dir+fileName+'.pkl', 'wb')
    pickle.dump(tasks, fobj)
    fobj.close()



def visualize_pickPlace(fileName, xRange = [-.1, .1], yRange=[0.6, 0.8]):

    tasks = np.array(read_goals(fileName))
    from matplotlib import pyplot as plt

    for i in range(len(tasks)):
        task = tasks[i]

        color = np.random.uniform(0,1, size=3)
        plt.annotate(xy = task['obj_init_pos'][:2], s= 'O'+str(i), color=color)
        plt.annotate(xy = task['goal'][:2], s='G'+str(i), color=color)


    plt.xlim(xRange[0], xRange[1])
    plt.ylim(yRange[0], yRange[1])


    plt.savefig(fileName+'.png')



def modify(oldName, newName):

    tasks = read_goals(oldName)


    new_tasks = []

    
    for old_task in tasks:

        new_task={}
       
        new_task['obj_init_pos'] = np.concatenate([old_task['obj_init_pos'], [0.02]])
        new_task['goal'] = np.concatenate([old_task['goal'], [0.02]]) 
        new_task['height'] = 0.1

        new_tasks.append(new_task)

    

    fobj = open(save_dir+newName+'.pkl', 'wb')
    pickle.dump(new_tasks, fobj)
    fobj.close()


gen_pickPlaceGoals('pickPlace_20X20_6_8')


visualize_pickPlace('pickPlace_20X20_6_8')

#read_goals('pickPlace_20X20_6_8')




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








