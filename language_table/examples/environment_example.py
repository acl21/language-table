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

"""Example for running the Language-Table environment."""

from collections.abc import Sequence

from absl import app

from language_table.environments import blocks
from language_table.environments import language_table_multigoals
from language_table.environments.rewards import (
    block2multicolours
)

from matplotlib import pyplot as plt
import numpy as np


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    env = language_table_multigoals.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_PURPLE,
        reward_factory=block2multicolours.BlockToMultiColoursReward,
        control_frequency=15.0,
        show_goals=True,
        goal_color="red-blue-green-yellow",
        render_text_in_image=False,
    )
    _ = env.reset()

    # Take a few random actions.
    for _ in range(5):
        env.step(np.array([0.0, 0.05]))

    # Save a rendered image.
    plt.imsave("language_table_render.png", env.render())


if __name__ == "__main__":
    app.run(main)
