import pickle
with open('data_2019-11-15T14-58-08.pkl', 'rb') as fp:
    trajectories = pickle.load(fp)

k = 1
for i in trajectories:
    if(i['rewards'][len(i['rewards'])-1] != 0):   
        k += 1

print(k / 2000)
