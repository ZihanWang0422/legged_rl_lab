# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 AMP environment registration."""

import gymnasium as gym

from . import agents

# Note: G1 specific config file needs to be created similar to go1_amp_env_cfg.py
# For now, we register a placeholder that you can implement later

# gym.register(
#     id="G1-AMP-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AMPEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#         "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
#     },
# )
