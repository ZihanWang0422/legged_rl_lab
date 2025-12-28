# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for AMP locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Time-based Terminations
# ==============================================================================


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when episode length exceeds maximum."""
    return env.episode_length_buf >= env.max_episode_length


# ==============================================================================
# Contact-based Terminations
# ==============================================================================


def illegal_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    Terminate when illegal body parts contact ground.
    Typically: base, thigh, calf (not feet).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.any(torch.norm(contact_forces[:, 0, :, :], dim=-1) > threshold, dim=1)
    return contact


# ==============================================================================
# Base State Terminations
# ==============================================================================


def base_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height drops below threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def base_orientation_limit(
    env: ManagerBasedRLEnv,
    roll_limit: float = 0.5,
    pitch_limit: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminate when base orientation exceeds limits.
    Uses projected gravity to compute roll/pitch.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_gravity = asset.data.projected_gravity_b
    
    # Approximate roll and pitch from projected gravity
    # gravity_b ≈ [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]
    roll = torch.atan2(-proj_gravity[:, 1], -proj_gravity[:, 2])
    pitch = torch.asin(torch.clamp(proj_gravity[:, 0], -1.0, 1.0))
    
    roll_violation = torch.abs(roll) > roll_limit
    pitch_violation = torch.abs(pitch) > pitch_limit
    
    return roll_violation | pitch_violation


def root_velocity_limit(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 10.0,
    max_ang_vel: float = 10.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root velocity exceeds limits (indicates instability)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    lin_vel_mag = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    ang_vel_mag = torch.norm(asset.data.root_ang_vel_w, dim=-1)
    
    return (lin_vel_mag > max_lin_vel) | (ang_vel_mag > max_ang_vel)


# ==============================================================================
# Joint State Terminations
# ==============================================================================


def joint_pos_out_of_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint position exceeds limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    limits = asset.data.soft_joint_pos_limits
    
    below_lower = torch.any(pos < limits[:, :, 0], dim=-1)
    above_upper = torch.any(pos > limits[:, :, 1], dim=-1)
    
    return below_lower | above_upper


def joint_vel_out_of_limit(
    env: ManagerBasedRLEnv,
    max_vel: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint velocity exceeds limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel) > max_vel, dim=-1)


# ==============================================================================
# Composite Terminations
# ==============================================================================


def bad_orientation_or_contact(
    env: ManagerBasedRLEnv,
    roll_limit: float = 0.5,
    pitch_limit: float = 0.5,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Combined termination for orientation violation or illegal contact."""
    orientation_bad = base_orientation_limit(env, roll_limit, pitch_limit, asset_cfg)
    contact_bad = illegal_contact(env, contact_threshold, sensor_cfg)
    return orientation_bad | contact_bad
