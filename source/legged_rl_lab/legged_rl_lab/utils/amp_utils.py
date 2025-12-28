# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for AMP locomotion tasks."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


# ==============================================================================
# Go1/A1 Kinematics Constants
# ==============================================================================

# Center of mass offset
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])

# Hip offsets from body center (for Go1/A1 style quadrupeds)
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.0],   # FL
    [0.183, -0.047, 0.0],  # FR
    [-0.183, 0.047, 0.0],  # RL
    [-0.183, -0.047, 0.0], # RR
]) + COM_OFFSET

# Link lengths
L_HIP = 0.08505  # Hip link length
L_UPPER = 0.2    # Upper leg length
L_LOWER = 0.2    # Lower leg length


# ==============================================================================
# Forward Kinematics Functions
# ==============================================================================


def foot_position_in_hip_frame(
    angles: torch.Tensor,
    l_hip_sign: int = 1,
) -> torch.Tensor:
    """
    Compute foot position in hip frame using forward kinematics.
    
    Args:
        angles: Joint angles [theta_ab, theta_hip, theta_knee] shape (N, 3)
        l_hip_sign: +1 for left legs, -1 for right legs
    
    Returns:
        Foot position in hip frame, shape (N, 3)
    """
    theta_ab = angles[:, 0]   # Abduction/adduction
    theta_hip = angles[:, 1]  # Hip flexion
    theta_knee = angles[:, 2] # Knee flexion
    
    # Leg distance (using law of cosines)
    leg_distance = torch.sqrt(
        L_UPPER**2 + L_LOWER**2 + 
        2 * L_UPPER * L_LOWER * torch.cos(theta_knee)
    )
    
    # Effective swing angle
    eff_swing = theta_hip + theta_knee / 2
    
    # Position in hip frame before ab/ad rotation
    off_x_hip = -leg_distance * torch.sin(eff_swing)
    off_z_hip = -leg_distance * torch.cos(eff_swing)
    off_y_hip = L_HIP * l_hip_sign
    
    # Apply ab/ad rotation
    off_x = off_x_hip
    off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
    off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
    
    return torch.stack([off_x, off_y, off_z], dim=-1)


def foot_positions_in_base_frame(
    joint_angles: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute all 4 foot positions in base frame.
    
    Args:
        joint_angles: All joint angles, shape (N, 12) 
                     Order: [FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...]
        device: Torch device
    
    Returns:
        Foot positions in base frame, shape (N, 12)
    """
    if device is None:
        device = joint_angles.device
    
    num_envs = joint_angles.shape[0]
    foot_positions = torch.zeros(num_envs, 12, device=device)
    
    # Process each leg
    for i in range(4):
        # Get joint angles for this leg
        leg_angles = joint_angles[:, i * 3:(i + 1) * 3]
        
        # Determine hip sign (left: +1, right: -1)
        # FL=0, FR=1, RL=2, RR=3 -> FL,RL are left (+1), FR,RR are right (-1)
        l_hip_sign = 1 if (i % 2 == 0) else -1
        
        # Compute foot position in hip frame
        foot_in_hip = foot_position_in_hip_frame(leg_angles, l_hip_sign)
        
        # Add hip offset to get position in base frame
        hip_offset = HIP_OFFSETS[i].to(device)
        foot_positions[:, i * 3:(i + 1) * 3] = foot_in_hip + hip_offset
    
    return foot_positions


# ==============================================================================
# AMP Observation Construction
# ==============================================================================


def build_amp_observations(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    root_height: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build AMP observation tensor.
    
    Args:
        joint_pos: Joint positions, shape (N, 12)
        joint_vel: Joint velocities, shape (N, 12)
        root_lin_vel: Root linear velocity in base frame, shape (N, 3)
        root_ang_vel: Root angular velocity in base frame, shape (N, 3)
        root_height: Root height (z position), shape (N,) or (N, 1)
    
    Returns:
        AMP observation tensor, shape (N, 43)
    """
    if device is None:
        device = joint_pos.device
    
    # Ensure root_height is 2D
    if root_height.dim() == 1:
        root_height = root_height.unsqueeze(-1)
    
    # Compute foot positions in base frame
    foot_pos = foot_positions_in_base_frame(joint_pos, device)
    
    # Concatenate all observations
    # Order: joint_pos(12), foot_pos(12), lin_vel(3), ang_vel(3), joint_vel(12), z(1)
    amp_obs = torch.cat([
        joint_pos,      # 12
        foot_pos,       # 12
        root_lin_vel,   # 3
        root_ang_vel,   # 3
        joint_vel,      # 12
        root_height,    # 1
    ], dim=-1)
    
    return amp_obs  # Total: 43


# ==============================================================================
# Motion Data Processing
# ==============================================================================


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion [w, x, y, z], shape (..., 4)
    
    Returns:
        Euler angles [roll, pitch, yaw], shape (..., 3)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1, 1)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw], shape (..., 3)
    
    Returns:
        Quaternion [w, x, y, z], shape (..., 4)
    """
    roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=-1)


# ==============================================================================
# Validation Utilities
# ==============================================================================


def validate_amp_obs_dim(
    num_joints: int = 12,
    expected_dim: int = 43,
) -> bool:
    """
    Validate AMP observation dimension.
    
    For Go1 (12 DOF):
    - joint_pos: 12
    - foot_pos: 12 (4 feet * 3)
    - base_lin_vel: 3
    - base_ang_vel: 3
    - joint_vel: 12
    - z_pos: 1
    Total: 43
    """
    computed_dim = num_joints + 12 + 3 + 3 + num_joints + 1
    return computed_dim == expected_dim


def print_amp_obs_structure(num_joints: int = 12):
    """Print AMP observation structure for debugging."""
    idx = 0
    print("AMP Observation Structure:")
    print(f"  [{idx}:{idx + num_joints}] joint_pos ({num_joints})")
    idx += num_joints
    print(f"  [{idx}:{idx + 12}] foot_pos (12)")
    idx += 12
    print(f"  [{idx}:{idx + 3}] base_lin_vel (3)")
    idx += 3
    print(f"  [{idx}:{idx + 3}] base_ang_vel (3)")
    idx += 3
    print(f"  [{idx}:{idx + num_joints}] joint_vel ({num_joints})")
    idx += num_joints
    print(f"  [{idx}:{idx + 1}] z_pos (1)")
    idx += 1
    print(f"  Total dimension: {idx}")
