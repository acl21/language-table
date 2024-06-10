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
from language_table.environments import language_table_new
from language_table.environments.rewards import (
    block2rglocation,
)
from language_table.utils.recorder import SimpleRecorder

from matplotlib import pyplot as plt
import numpy as np
import time
import cv2
import pybullet as p


# Function to handle key presses


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    env = language_table_new.LanguageTableNew(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_B,
        reward_factory=block2rglocation.BlockToRGLocationReward,
        control_frequency=15.0,
        show_goals=True,
        render_text_in_image=False,
        show_gui=True,
        goal_color="red"
    )
    _ = env.reset()
    
    # Data folder
    data_folder = "data"
    day = time.strftime("%Y-%m-%d")
    time_ = time.strftime("%H-%M-%S")
    data_folder = f"{data_folder}/{day}/{time_}"

    # Recorder
    recorder = SimpleRecorder(save_dir=data_folder)

    recording = False
    action_key_pressed = False
    delete_last_episode = False
    try:
        while True:
            keys = p.getKeyboardEvents()
            (x, y), action_key_pressed = get_action_keyboard(keys)

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
                    raise ValueError("Cannot reset while recording")
            
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

def get_action_keyboard(keys):
    action_key_pressed = False
    x, y = 0, 0
    # Arrow keys to move the arm
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW]:
        y = -0.01
        action_key_pressed = True
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW]:
        y = 0.01
        action_key_pressed = True
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW]:
        x = -0.01
        action_key_pressed = True
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW]:
        x = 0.01
        action_key_pressed = True
    return (x, y), action_key_pressed


if __name__ == "__main__":
    app.run(main)
