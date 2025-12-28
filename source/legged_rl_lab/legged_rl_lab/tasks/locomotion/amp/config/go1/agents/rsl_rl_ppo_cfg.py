# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL AMP configuration for Go1 robot."""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go1AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for Go1 AMP training using RSL-RL."""
    
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "go1_amp"
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go1AmpPPORunnerFlatCfg(Go1AmpPPORunnerCfg):
    """Configuration for Go1 AMP training on flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Reduce network size for flat terrain
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [256, 128, 64]
        
        # Fewer iterations needed
        self.max_iterations = 20000
        self.experiment_name = "go1_amp_flat"


@configclass
class Go1AmpPPORunnerRoughCfg(Go1AmpPPORunnerCfg):
    """Configuration for Go1 AMP training on rough terrain."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # More exploration for rough terrain
        self.policy.init_noise_std = 1.2
        self.algorithm.entropy_coef = 0.015
        
        # More iterations for complex terrain
        self.max_iterations = 80000
        self.experiment_name = "go1_amp_rough"

