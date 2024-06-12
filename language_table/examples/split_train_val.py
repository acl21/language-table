import shutil
import numpy as np

FILENAME = "episode"
TRAIN = "training"
VAL = "validation"
N_DIGITS = 6
VAL_EPISODES = 50

data_dir = "/export/home/lagandua/language-table/data/Push2Green/"
ep_info = np.load(f"{data_dir}/{TRAIN}/ep_start_end_ids.npy")


# Choose random epsiodes for validation
np.random.seed(0)
val_episodes = np.random.choice(ep_info.shape[0], VAL_EPISODES, replace=False)
train_episodes = np.array([i for i in range(ep_info.shape[0]) if i not in val_episodes])

# Move them to validation folder
for val_id in val_episodes:
    start_idx, end_idx = ep_info[val_id]
    for idx in range(start_idx, end_idx+1):
        idx_str = str(idx).zfill(N_DIGITS)
        shutil.move(f"{data_dir}/{TRAIN}/{FILENAME}_{idx_str}.npz", f"{data_dir}/{VAL}/{FILENAME}_{idx_str}.npz")

# save the train and val episodes ids
np.save(f"{data_dir}/{TRAIN}/ep_start_end_ids.npy", ep_info[train_episodes])
np.save(f"{data_dir}/{VAL}/ep_start_end_ids.npy", ep_info[val_episodes])