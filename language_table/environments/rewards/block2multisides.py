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

"""Defines block2multisides reset and reward."""
import enum

from typing import Any, List
from absl import logging
from language_table.environments import blocks as blocks_module
from language_table.environments.rewards import reward as base_reward
from language_table.environments.rewards import synonyms
from language_table.environments.rewards import task_info
import numpy as np


# There's a small offset in the Y direction to subtract.
# The red dots represent the bounds of the arm, which are not exactly in the
# center of the boards.
# This should only matter for this reward, which deals with absolute locations.
X_BUFFER = 0.025
# X_BUFFER = 0

X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
X_MIN = X_MIN_REAL - X_BUFFER
X_MAX = X_MAX_REAL - X_BUFFER
Y_MIN = Y_MIN_REAL
Y_MAX = Y_MAX_REAL
CENTER_X = (X_MAX - X_MIN) / 2.0 + X_MIN
CENTER_Y = (Y_MAX - Y_MIN) / 2.0 + Y_MIN

BLOCK2SIDE_TARGET_DISTANCE = 0.05


class Locations(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    

ABSOLUTE_LOCATIONS = {
    "left": [-1, -0.2],
    "right": [1, 0.2],
}

LOCATION_SYNONYMS = {
    "left": ["left side of the board", "left", "left side"],
    "right": ["right side of the board", "right", "right side"],
    "left-right": ["either left or right side of the board", "either left or right", "either left or right sides"],
}
L_GOAL = "left"
R_GOAL = "right"
LR_GOAL = "left-right"

BLOCK2SIDES_VERBS = [
    "move the",
    "push the",
    "slide the",CENTER_X
]


def generate_all_instructions(block_mode):
    """Generate all instructions for block2relativeposition."""
    all_instructions = []
    all_block_text_descriptions = blocks_module.get_blocks_text_descriptions(block_mode)
    for block_text in all_block_text_descriptions:
        for location in ABSOLUTE_LOCATIONS:
            for location_syn in LOCATION_SYNONYMS[location]:
                for verb in BLOCK2SIDES_VERBS:
                    # Add instruction.
                    inst = f"{verb} {block_text} to the {location_syn}"
                    all_instructions.append(inst)
    return all_instructions


class BlockToMultiSidesReward(base_reward.LanguageTableReward):
    """Calculates reward/instructions for 'push block to multi colours'."""

    def __init__(self, goal_reward, rng, delay_reward_steps, block_mode):
        super(BlockToMultiSidesReward, self).__init__(
            goal_reward=goal_reward,
            rng=rng,
            delay_reward_steps=delay_reward_steps,
            block_mode=block_mode,
        )
        self._block = None
        self._instruction = None
        self._location = None
        self._target_translation = None
        self._goal = None
        self._goal_str = None
        self._multi_goal = False

    def _sample_instruction(self, block, blocks_on_table):
        """Randomly sample a task involving two objects."""

        verb = self._rng.choice(synonyms.PUSH_VERBS)
        # Get some synonym for block.
        block_text = self._rng.choice(
            synonyms.get_block_synonyms(block, blocks_on_table)
        )
        # Get some synonym for location.
        location_syn = self._rng.choice(LOCATION_SYNONYMS[self._goal])
        return f"{verb} {block_text} to the {location_syn}"

    def reset(self, state, blocks_on_table):
        """Chooses new target block and location."""
        block = self._sample_object(blocks_on_table)

        location = self._goal

        info = self.reset_to(state, block, location, blocks_on_table)
        # If the state of the board already triggers the reward, try to reset
        # again with a new configuration.
        if self._multi_goal:
            for target in self._target_translation:
                if self._in_goal_region_start(state, block, target):
                    # Try again with a new board configuration.
                    return task_info.FAILURE
        else:
            if self._in_goal_region_start(state, self._block, self._target_translation):
                # Try again with a new board configuration.
                return task_info.FAILURE
        return info

    def reset_to(self, state, block, location, blocks_on_table):
        """Reset to a particular task definition."""
        self._block = block

        # Sample an instruction.
        self._instruction = self._sample_instruction(block, blocks_on_table)

        # Get the corresponding target_translation.
        if self._multi_goal:
            target_translation = []
            for goal in self._goal.split("-"):
                target_translation.append(np.array(ABSOLUTE_LOCATIONS[goal]))
        else:
            target_translation = np.array(ABSOLUTE_LOCATIONS[location])

        # Cache the target location corresponding to the instruction.
        self._target_translation = np.copy(target_translation)
        self._location = location
        info = self.get_current_task_info(state)
        self._in_reward_zone_steps = 0
        return info

    @property
    def target_translation(self):
        return self._target_translation

    def reward(self, state):
        """Calculates reward given state."""
        info = -1
        if self._multi_goal:
            reward, done = self.reward_for(
                state, self._block, self._target_translation[0]
            )
            if reward > 0:
                info = 0
            for idx, target in enumerate(self._target_translation[1:]):
                reward_, done_ = self.reward_for(state, self._block, target)
                if reward_ > 0:
                    info = idx + 1
                reward = max(reward, reward_)
                done = done or done_
            if info < 0:
                info = ""
            else:
                info = self._goal_str.split("-")[info]
        else:
            info = ""
            (
                reward,
                done,
            ) = self.reward_for(state, self._block, self._target_translation)
        return reward, done, info

    def reward_for(self, state, pushing_block, target_translation):
        """Returns 1. if pushing_block is in location."""
        reward = 0.0
        done = False

        in_goal_region = self._in_goal_region(state, pushing_block, target_translation)

        if in_goal_region:
            if self._in_reward_zone_steps >= self._delay_reward_steps:
                reward = self._goal_reward
                done = True
            else:
                logging.info("In reward zone for %d steps", self._in_reward_zone_steps)
                self._in_reward_zone_steps += 1
        return reward, done

    def _in_goal_region(self, state, pushing_block, target_translation):
        # Get current location of the target block.
        current_translation, _ = self._get_pose_for_block(pushing_block, state)
        
        if target_translation[0] > 0: # Right side.
            return current_translation[1] > target_translation[1]
        else: # Left side.
            return current_translation[1] < target_translation[1]

    def _in_goal_region_start(self, state, pushing_block, target_translation):
        """
        To keeps the blocks far from the goal location at the start.
        """
        # Get current location of the target block.
        current_translation, _ = self._get_pose_for_block(pushing_block, state)
        
        # Check block's Y-axis. Return false when they are in goal region.
        if target_translation[0] > 0: # Right side.
            return current_translation[1] > target_translation[1]
        else: # Left side.
            return current_translation[1] < target_translation[1]

        return True

    def get_goal_region(self):
        """Returns the (target translation, radius) tuple for red and green goals."""
        return self._goal, BLOCK2SIDE_TARGET_DISTANCE

    def reward_for_info(self, state, info):
        return self.reward_for(state, info.block, info.target_translation)

    def debug_info(self, state):
        """Returns 1. if pushing_block is in location."""
        # Get current location of the target block.
        current_translation, _ = self._get_pose_for_block(self._block, state)
        # Compute distance between current translation and target.
        dist = np.linalg.norm(
            np.array(current_translation) - np.array(self._target_translation)
        )
        return dist

    def get_current_task_info(self, state):
        return task_info.Block2MultiColoursTaskInfo(
            instruction=self._instruction,
            block=self._block,
            location=self._location,
            target_translation=self._target_translation,
        )

    def set_goal_from_text(self, str):
        self._goal_str = str
        goal_list = str.split("-")
        l_flag, r_flag, = False, False
        if "left" in goal_list:
            l_flag = True
            self._goal = L_GOAL
        if "right" in goal_list:
            r_flag = True
            self._goal = R_GOAL
        if l_flag and r_flag:
            self._goal = LR_GOAL

        self._multi_goal = len(self._goal.split("-")) > 1
