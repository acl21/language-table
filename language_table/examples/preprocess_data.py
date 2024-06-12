import argparse
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

FILENAME = "episode"
N_DIGITS = 6
TRAIN = "training"
VAL = "validation"

def listdirs(rootdir):
    list_dirs = []
    for path in Path(rootdir).iterdir():
        if path.is_dir():
            if (path / "ep_start_end_ids.npy").is_file():
                list_dirs.append(path)
            else:
                result = listdirs(path)
                if result:
                    list_dirs.append(result)
    if list_dirs:
        return list_dirs

def get_frame(path, i):
    filename = Path(path) / f"frame_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def get_filename(path, subset, i):
    return Path(path) / subset / f"{FILENAME}_{i:0{N_DIGITS}d}.npz"

def process_data(recording_dir, i):
    data = get_frame(recording_dir, i)
    pos = data["effector_translation"]
    rgb = data["rgb"]
    action = data["action"]
    reward = data["rew"]
    done = data["done"]
    state = data["info"][()]["state"]
    save_data = {
        "pos": pos,
        "rgb": rgb,
        "action": action,
        "reward": reward,
        "done": done,
        "state": state,
    }
    return save_data


def create_dataset(recording_dirs, output_dir):
    ep_start_end_ids_all = []
    new_end_idx = 0
    for recording_dir in tqdm(recording_dirs):
        ep_start_end_ids = np.sort(np.load(recording_dir / "ep_start_end_ids.npy"))
        for start_idx, end_idx in tqdm(ep_start_end_ids, leave=False):
            new_start_idx = new_end_idx
            for i in range(start_idx + 1, end_idx):
                save_data = process_data(recording_dir, i)
                np.savez_compressed(get_filename(output_dir, TRAIN, new_end_idx), **save_data)
                new_end_idx += 1
            ep_start_end_ids_all.append((new_start_idx, new_end_idx - 1))
    np.save(output_dir / TRAIN / "ep_start_end_ids.npy", ep_start_end_ids_all)

def main(args):
    list_recording_dirs = listdirs(args.data_path)
    if len(list_recording_dirs) == 1:
        recording_dirs = list_recording_dirs
    else:
        # TODO: Write a function to address this automatically
        # Use this line if the data is collected over multiple days
        # recording_dirs = [item for sublist in list_recording_dirs for item in sublist]
        
        # Use this line if the data is collected in a single day
        recording_dirs = [item for item in list_recording_dirs]
    print("Found following subfolders containing recordings: ", recording_dirs)
    output_dir = Path(args.output_path)
    if output_dir.exists():
        print(
            f"The output dir {str(output_dir)} already exists. Do you want to overwrite? (Y/N)"
        )
        c = input()
        if c == "Y" or c == "y":
            os.rmdir(output_dir)
        elif c == "N" or c == "n":
            return
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / TRAIN, exist_ok=True)
    os.makedirs(output_dir / VAL, exist_ok=True)
    create_dataset(recording_dirs, output_dir)

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="") 
    args = parser.parse_args()
    main(args)