# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for AMP locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Velocity Tracking Rewards
# ==============================================================================


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward tracking of linear velocity commands (xy axes) using exponential kernel.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward tracking of angular velocity commands (yaw) using exponential kernel.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# ==============================================================================
# Base State Penalties
# ==============================================================================


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation from target."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


# ==============================================================================
# Joint State Penalties
# ==============================================================================


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions close to limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    pos_limits = asset.data.soft_joint_pos_limits
    
    # Compute soft limits
    limit_range = pos_limits[:, :, 1] - pos_limits[:, :, 0]
    limit_mid = (pos_limits[:, :, 1] + pos_limits[:, :, 0]) / 2
    soft_lower = limit_mid - soft_ratio * limit_range / 2
    soft_upper = limit_mid + soft_ratio * limit_range / 2
    
    # Penalize positions outside soft limits
    out_of_limits = torch.zeros_like(pos)
    out_of_limits += (soft_lower - pos).clamp(min=0.0)
    out_of_limits += (pos - soft_upper).clamp(min=0.0)
    
    return torch.sum(out_of_limits, dim=1)


def joint_pos_target_l2(
    env: ManagerBasedRLEnv,
    target: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


# ==============================================================================
# Action Penalties
# ==============================================================================


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (change in actions)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large actions."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


# ==============================================================================
# Contact / Collision Penalties
# ==============================================================================


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize undesired body contacts (thigh, calf, base, etc.)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.any(torch.norm(contact_forces[:, 0, :, :], dim=-1) > threshold, dim=1)
    return contact.float()


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    horizontal_force = torch.norm(forces[:, :, :2], dim=-1)
    vertical_force = torch.abs(forces[:, :, 2])
    return torch.any(horizontal_force > 5.0 * vertical_force, dim=1).float()


# ==============================================================================
# Feet / Gait Rewards
# ==============================================================================


def feet_air_time(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward long steps (feet air time).
    Only rewards on first contact after air time > threshold.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact state
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact = contact_forces[:, 0, :] > 1.0
    
    # Get command magnitude
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    
    # Initialize air time buffer if needed
    if not hasattr(env, "_feet_air_time"):
        env._feet_air_time = torch.zeros(env.num_envs, 4, device=env.device)
        env._last_contacts = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
    
    # Update air time
    contact_filt = torch.logical_or(contact, env._last_contacts)
    first_contact = (env._feet_air_time > 0.0) * contact_filt
    env._feet_air_time += env.step_dt
    
    # Reward for air time
    rew_air_time = torch.sum((env._feet_air_time - threshold) * first_contact, dim=1)
    rew_air_time *= moving  # No reward for zero command
    
    # Reset air time for contacts
    env._feet_air_time *= ~contact_filt
    env._last_contacts = contact
    
    return rew_air_time


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize motion at zero commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    standing = torch.norm(command[:, :2], dim=1) < 0.1
    joint_deviation = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return joint_deviation * standing


# ==============================================================================
# Survival / Episode Rewards
# ==============================================================================


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for termination."""
    return env.termination_manager.terminated.float()
