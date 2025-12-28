# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for AMP locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==============================================================================
# Base Observations (Policy Input)
# ==============================================================================


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity direction projected to base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Velocity commands [vx, vy, omega_z]."""
    return env.command_manager.get_command("base_velocity")


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last action applied to the environment."""
    return env.action_manager.action


# ==============================================================================
# AMP Observations (Discriminator Input)
# ==============================================================================


def amp_observations(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    AMP observations for discriminator.
    Includes: joint_pos, foot_pos_in_base, base_lin_vel, base_ang_vel, joint_vel, z_pos
    
    Total dim for quadruped (12 DOF):
    - joint_pos: 12
    - foot_pos_in_base: 12 (4 feet * 3)
    - base_lin_vel: 3
    - base_ang_vel: 3
    - joint_vel: 12
    - z_pos: 1
    Total: 43
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Joint positions
    joint_pos = asset.data.joint_pos
    
    # Foot positions in base frame (需要根据具体机器人实现)
    # 这里提供一个占位实现，你需要根据实际机器人运动学调整
    foot_pos = _compute_foot_positions_in_base_frame(asset)
    
    # Base velocities in base frame
    base_lin_vel = asset.data.root_lin_vel_b
    base_ang_vel = asset.data.root_ang_vel_b
    
    # Joint velocities
    joint_vel = asset.data.joint_vel
    
    # Root height (z position)
    z_pos = asset.data.root_pos_w[:, 2:3]
    
    return torch.cat([joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos], dim=-1)


def _compute_foot_positions_in_base_frame(asset: Articulation) -> torch.Tensor:
    """
    Compute foot positions in base frame using forward kinematics.
    This is a placeholder - you need to implement actual FK for your robot.
    
    For Go1/A1 type robots:
    - 4 legs, each with 3 joints (hip, thigh, calf)
    - Returns [num_envs, 12] tensor (4 feet * 3 xyz)
    """
    # 占位实现 - 你需要根据实际机器人运动学计算
    # 可以使用 asset.data.body_pos_w 获取刚体位置，然后转换到 base frame
    num_envs = asset.data.joint_pos.shape[0]
    device = asset.data.joint_pos.device
    
    # TODO: 实现实际的正运动学计算
    # 示例：使用刚体位置（如果机器人有足端刚体）
    # foot_body_ids = [4, 8, 12, 16]  # 根据URDF确定
    # foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]
    # 转换到 base frame...
    
    # 临时返回零向量，需要替换为实际实现
    return torch.zeros(num_envs, 12, device=device)


# ==============================================================================
# Height Measurements (Optional)
# ==============================================================================


def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """
    Height scan around the robot.
    Returns height relative to robot base.
    """
    # 如果有高度扫描传感器
    # sensor = env.scene.sensors[sensor_cfg.name]
    # return sensor.data.ray_hits_w[..., 2] - env.scene["robot"].data.root_pos_w[:, 2:3]
    
    # 占位实现
    num_envs = env.num_envs
    device = env.device
    return torch.zeros(num_envs, 187, device=device)  # 典型的高度点数量


# ==============================================================================
# Contact States
# ==============================================================================


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Binary foot contact states.
    Returns 1 if contact force > threshold, else 0.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact = torch.norm(contact_forces[:, 0, :, :], dim=-1) > threshold
    return contact.float()
