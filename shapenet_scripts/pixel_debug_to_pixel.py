import pickle
import os

BUFFER_NAME = (
    '/media/avi/data/Work/github/jannerm/bullet-manipulation/data/sac_runs_feb11/'
    'env-11Feb2020-af888d54-seed=527_2020-02-11_20-11-04uk2zf5y4/pixels_buffer/'
    'consolidated_buffer.pkl')

if __name__ == "__main__":

    with open(BUFFER_NAME, 'rb') as f:
        buffer_from_disk = pickle.load(f)

    dims_to_keep = 4
    for i in range(len(buffer_from_disk['observations'])):
        buffer_from_disk['observations'][i][0]['state'] = \
            buffer_from_disk['observations'][i][0]['state'][:dims_to_keep]
        buffer_from_disk['next_observations'][i][0]['state'] = \
            buffer_from_disk['next_observations'][i][0]['state'][:dims_to_keep]

    savefolder = os.path.dirname(BUFFER_NAME)
    savepath = os.path.join(savefolder, 'consolidated_buffer_pixel_only.pkl')
    pickle.dump(buffer_from_disk, open(savepath, 'wb+'))