# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) locomotion tasks."""

from .amp_env_cfg import AMPCfg, AMPEnvCfg, AMPSceneCfg
# from legged_rl_lab.utils.amp_utils import (
#     build_amp_observations,
#     foot_position_in_hip_frame,
#     foot_positions_in_base_frame,
# )

# Import robot-specific configs to trigger gym registration
from .config.go1 import *  # noqa: F401, F403
# from .config.g1 import *  # noqa: F401, F403  # Uncomment when G1 config is ready
