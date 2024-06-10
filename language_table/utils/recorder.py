import os
from pathlib import Path
import json
import datetime
import numpy as np
from PIL import Image
import multiprocessing as mp
import threading
import logging
import time

log = logging.getLogger(__name__)

def count_previous_frames():
    return len(list(Path.cwd().glob("frame*.npz")))


class SimpleRecorder:
    def __init__(self, save_dir="", n_digits=6):
        """
        SimpleRecorder is a recorder to save frames with a simple step function.
        Recordings can be loaded with load_rec_list/PlaybackEnv.

        Arguments:
            save_dir: Directory in which to save
            n_digits: zero padding for files
            save_images: save .jpg image files as well
        """
        self.recording = False
        self.queue = []
        self.save_dir = save_dir
        self.save_frame_cnt = count_previous_frames()
        self.current_episode_filenames = []
        self.n_digits = n_digits
        self.delete_thread = None
        self.ep_start_end_ids = []
        self.running = True
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, action, obs, rew, done, info):
        """
        Save the data every step.

        Args:
            action: Action used to command the robot.
            obs: Env observation.
            rew: Env reward.
            done: Env done.
            info:  Env info.
        """
        filename = f"frame_{self.save_frame_cnt:0{self.n_digits}d}.npz"
        filename = os.path.join(self.save_dir, filename)
        self.current_episode_filenames.append(filename)
        self.save_frame_cnt += 1
        self.queue.append((filename, action, obs, rew, done, info))

    def _process_queue(self):
        """
        Process function for queue.
        """
        for msg in self.queue:
            filename, action, obs, rew, done, info = msg
            np.savez_compressed(filename, **obs, action=action, rew=rew, done=done, info=info)
        self.queue = []


    def _save(self):
        self._process_queue()
        np.save(f"{self.save_dir}/ep_start_end_ids.npy", self.ep_start_end_ids)
    
    def delete_last_episode(self):
        self.delete_thread = threading.Thread(target=self._delete_last_episode, daemon=True)
        self.delete_thread.start()
        self.ep_start_end_ids = self.ep_start_end_ids[:-1]
        np.save(f"{self.save_dir}/ep_start_end_ids.npy", self.ep_start_end_ids)

    def _delete_last_episode(self):
        log.info("Delete episode")
        while not (len(self.queue) == 0):
            log.info("Wait until files are saved")
            time.sleep(0.01)
        num_frames = len(self.current_episode_filenames)
        log.info(f"Deleting last episode with {num_frames} frames")
        for filename in self.current_episode_filenames:
            os.remove(filename)
        log.info("Finished deleting")
        self.save_frame_cnt -= num_frames
        self.current_episode_filenames = []
    
    def start_recording(self):
        self.recording = True
        self.current_episode_filenames = []
        self.ep_start_end_ids.append([self.save_frame_cnt])
        log.info(f"Start recording episode {len(self.ep_start_end_ids)}")
    
    def stop_recording(self):
        self.recording = False
        self.ep_start_end_ids[-1].append(self.save_frame_cnt)
        self._save()
        np.save(f"{self.save_dir}/ep_start_end_ids.npy", self.ep_start_end_ids)
        log.info(f"Stop recording episode {len(self.ep_start_end_ids)}")