# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for AMP locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Command Curriculum
# ==============================================================================


def modify_velocity_command_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    min_step: float = 0.5,
    max_velocity: float = 2.0,
    tracking_reward_name: str = "track_lin_vel_xy_exp",
    tracking_threshold: float = 0.8,
) -> torch.Tensor:
    """
    Curriculum that increases velocity command range based on tracking performance.
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs being reset.
        term_name: Name of the term in command manager.
        min_step: Minimum step for increasing range.
        max_velocity: Maximum velocity to reach.
        tracking_reward_name: Name of tracking reward for evaluation.
        tracking_threshold: Threshold for considering tracking successful.
    
    Returns:
        Current velocity range.
    """
    # Get current command range
    command_term = env.command_manager.get_term(term_name)
    current_range = command_term.cfg.ranges.lin_vel_x
    
    # Check if we should update curriculum
    if len(env_ids) == 0:
        return torch.tensor([current_range[0], current_range[1]], device=env.device)
    
    # Get tracking performance (average reward)
    if hasattr(env, "episode_sums") and tracking_reward_name in env.episode_sums:
        avg_tracking = torch.mean(env.episode_sums[tracking_reward_name][env_ids])
        max_episode_length = env.max_episode_length
        
        # Normalize by episode length
        normalized_tracking = avg_tracking / max_episode_length
        
        # If tracking is good, increase range
        if normalized_tracking > tracking_threshold:
            new_min = max(current_range[0] - min_step, -max_velocity)
            new_max = min(current_range[1] + min_step, max_velocity)
            command_term.cfg.ranges.lin_vel_x = (new_min, new_max)
    
    return torch.tensor([current_range[0], current_range[1]], device=env.device)


# ==============================================================================
# Terrain Curriculum
# ==============================================================================


def terrain_levels_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    move_up_distance: float = 1.0,
    move_down_distance: float = 0.5,
) -> torch.Tensor:
    """
    Terrain curriculum that adjusts difficulty based on distance traveled.
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs being reset.
        move_up_distance: Distance threshold to increase terrain level.
        move_down_distance: Distance threshold to decrease terrain level.
    
    Returns:
        Updated terrain levels.
    """
    if len(env_ids) == 0:
        return env.terrain_levels if hasattr(env, "terrain_levels") else torch.zeros(0)
    
    # Initialize terrain levels if not exist
    if not hasattr(env, "terrain_levels"):
        env.terrain_levels = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.max_terrain_level = 10  # Default max level
    
    # Calculate distance traveled
    asset = env.scene["robot"]
    current_pos = asset.data.root_pos_w[env_ids, :2]
    
    if hasattr(env, "_reset_positions"):
        start_pos = env._reset_positions[env_ids]
        distance = torch.norm(current_pos - start_pos, dim=-1)
        
        # Update levels
        move_up = distance > move_up_distance
        move_down = distance < move_down_distance
        
        env.terrain_levels[env_ids] += move_up.long() - move_down.long()
        env.terrain_levels[env_ids] = torch.clamp(env.terrain_levels[env_ids], 0, env.max_terrain_level)
    
    return env.terrain_levels


# ==============================================================================
# Reward Curriculum
# ==============================================================================


def modify_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight_range: tuple[float, float] = (0.0, 1.0),
    num_steps_to_max: int = 10000,
) -> torch.Tensor:
    """
    Curriculum that gradually increases a reward weight.
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs.
        term_name: Name of the reward term.
        weight_range: (min_weight, max_weight) range.
        num_steps_to_max: Number of environment steps to reach max weight.
    
    Returns:
        Current weight value.
    """
    # Calculate progress
    total_steps = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    progress = min(total_steps / num_steps_to_max, 1.0)
    
    # Linear interpolation
    current_weight = weight_range[0] + progress * (weight_range[1] - weight_range[0])
    
    # Update reward weight
    reward_term = env.reward_manager.get_term(term_name)
    if reward_term is not None:
        reward_term.weight = current_weight
    
    return torch.tensor([current_weight], device=env.device)


# ==============================================================================
# Push Curriculum
# ==============================================================================


def modify_push_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_push_vel: float = 0.5,
    max_push_vel: float = 2.0,
    num_steps_to_max: int = 50000,
) -> torch.Tensor:
    """
    Curriculum that gradually increases external push velocity.
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs.
        initial_push_vel: Initial maximum push velocity.
        max_push_vel: Maximum push velocity to reach.
        num_steps_to_max: Steps to reach max push velocity.
    
    Returns:
        Current max push velocity.
    """
    total_steps = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    progress = min(total_steps / num_steps_to_max, 1.0)
    
    current_push_vel = initial_push_vel + progress * (max_push_vel - initial_push_vel)
    
    # Store for use in push event
    env._max_push_vel_xy = current_push_vel
    
    return torch.tensor([current_push_vel], device=env.device)


# ==============================================================================
# AMP-specific Curriculum
# ==============================================================================


def amp_task_reward_lerp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    initial_lerp: float = 0.0,
    final_lerp: float = 0.5,
    num_steps_to_final: int = 100000,
) -> torch.Tensor:
    """
    Curriculum for AMP task reward interpolation.
    Controls balance between style (discriminator) and task rewards.
    
    lerp = 0: Pure style reward (only discriminator)
    lerp = 1: Pure task reward (only tracking, etc.)
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs.
        initial_lerp: Initial lerp value (typically 0 for pure style).
        final_lerp: Final lerp value.
        num_steps_to_final: Steps to reach final lerp.
    
    Returns:
        Current lerp value.
    """
    total_steps = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    progress = min(total_steps / num_steps_to_final, 1.0)
    
    current_lerp = initial_lerp + progress * (final_lerp - initial_lerp)
    
    # Store for use in runner/algorithm
    env._amp_task_reward_lerp = current_lerp
    
    return torch.tensor([current_lerp], device=env.device)
