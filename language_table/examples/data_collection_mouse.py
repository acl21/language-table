# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example for running the new Language-Table environment and collecting data."""

from collections.abc import Sequence

from absl import app

from language_table.environments import blocks
from language_table.environments import language_table_4goals
from language_table.environments.rewards import (
    block2colours,
)
from language_table.utils.recorder import SimpleRecorder

from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import pybullet as p
from pynput import mouse
import threading

# Global variable to store mouse position
mouse_position = [0, 0]

def on_move(x, y):
    global mouse_position
    mouse_position = [x, y]

# Function to start the mouse listener
def start_mouse_listener():
    with mouse.Listener(on_move=on_move) as listener:
        listener.join()

# Start the mouse listener in a separate thread
mouse_listener_thread = threading.Thread(target=start_mouse_listener)
mouse_listener_thread.start()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    goal_color = "red"
    env = language_table_4goals.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_PURPLE,
        reward_factory=block2colours.BlockToColoursReward,
        control_frequency=15.0,
        show_goals=True,
        render_text_in_image=False,
        show_gui=True,
        goal_color=goal_color
    )
    _ = env.reset()
    
    # Data folder
    data_folder = f"data/push2{goal_color}"
    day = time.strftime("%Y-%m-%d")
    time_ = time.strftime("%H-%M-%S")
    data_folder = f"{data_folder}/{day}/{time_}"

    # Recorder
    recorder = SimpleRecorder(save_dir=data_folder)

    recording = False
    action_key_pressed = False
    delete_last_episode = False
    prev_mouse_position = None
    try:
        while True:
            keys = p.getKeyboardEvents()

            if recording:
                if prev_mouse_position is None:
                    prev_mouse_position = mouse_position
                else:
                    y, x = np.clip((np.array(mouse_position) - np.array(prev_mouse_position)) / 10, -0.005, 0.005)
                    prev_mouse_position = mouse_position
            
            # Action is applied when mouse only when "m" is pressed
            if ord("m") in keys and keys[ord("m")] and p.KEY_WAS_RELEASED:
                action_key_pressed = True
            
            # Space to start recording
            if p.B3G_SPACE in keys and keys[p.B3G_SPACE] and p.KEY_WAS_RELEASED:
                recording = True
            
            # Shift to stop recording
            # if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] and p.KEY_WAS_RELEASED:
            #     if recording:
            #         recording = False
            
            # Delete to delete the last episode
            if ord("x") in keys and keys[ord("x")] and p.KEY_WAS_RELEASED:
                delete_last_episode = True
            
            # End to reset the environment
            if ord("r") in keys and keys[ord("r")] and p.KEY_WAS_RELEASED:
                if not recording:
                    env.reset()
                else:
                    recorder.stop_recording()
                    recording = False
                    recorder.delete_last_episode()
                    env.reset()
            
            # Do the needful only when action key "m" is pressed
            if action_key_pressed:
                if delete_last_episode:
                    recorder.delete_last_episode()
                    delete_last_episode = False
                if recording and not recorder.recording:
                    recorder.start_recording()
                if not recording and recorder.recording:
                    recorder.stop_recording()
                action_key_pressed = False
                obs, reward, done, info = env.step(np.array([x, y]))
                if recording and recorder.recording:
                    recorder.step(np.array([x, y]), obs, reward, done, info)
                if done:
                    if recording and recorder.recording:
                        recorder.stop_recording()
                        recording = False
                    env.reset()

    except KeyboardInterrupt:
        env.close()
        recorder._save()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)
