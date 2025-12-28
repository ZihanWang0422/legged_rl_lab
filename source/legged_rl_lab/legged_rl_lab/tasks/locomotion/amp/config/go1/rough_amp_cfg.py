# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go1 AMP environment configuration."""

from __future__ import annotations

import glob

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import (
    AMPCfg,
    AMPEnvCfg,
    AMPSceneCfg,
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from legged_rl_lab.tasks.locomotion.amp import mdp

##
# Pre-defined robot config
##
# 你需要导入或定义 Go1 的 ArticulationCfg
# from isaaclab_assets.robots.unitree import GO1_CFG
# 或者自定义:

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


GO1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/wzh/amp/legged_rl_lab/assets/robots/go1/go1.usd",
        # 如果你只有 URDF，请改用：
        # usd_path=None,
        # urdf_path="/home/wzh/amp/legged_rl_lab/assets/robots/go1/urdf/go1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FR_thigh_joint": 0.9,
            "RL_thigh_joint": 0.9,
            "RR_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "FR_calf_joint": -1.8,
            "RL_calf_joint": -1.8,
            "RR_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=80.0,
            damping=1.0,
        ),
    },
)


##
# Scene Configuration
##


@configclass
class Go1AMPSceneCfg(AMPSceneCfg):
    """Go1 AMP scene configuration."""

    robot: ArticulationCfg = GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# Actions Configuration
##


@configclass
class Go1ActionsCfg(ActionsCfg):
    """Go1 action configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.25,
        use_default_offset=True,
    )


##
# Observations Configuration
##


@configclass
class Go1ObservationsCfg(ObservationsCfg):
    """Go1 observation configuration."""
    pass  # Use base config, override if needed


##
# Events Configuration
##


@configclass
class Go1EventCfg(EventCfg):
    """Go1 event configuration with domain randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.75),
            "dynamic_friction_range": (0.25, 1.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )


##
# Rewards Configuration
##


@configclass
class Go1RewardsCfg(RewardsCfg):
    """Go1 reward configuration.
    
    Note: For AMP, most rewards are zeroed out since the discriminator
    provides the main learning signal. Only velocity tracking rewards
    are kept to guide toward commanded behavior.
    """

    # Velocity tracking with AMP-appropriate scaling
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5 / (0.005 * 6),  # Scaled by dt
        params={"std": 0.25, "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5 / (0.005 * 6),
        params={"std": 0.25, "command_name": "base_velocity"},
    )

    # All other rewards zeroed for pure AMP
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=0.0, params={"target_height": 0.25})
    torques = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=0.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.0)
    termination = RewTerm(func=mdp.is_terminated, weight=0.0)


##
# Terminations Configuration
##


@configclass
class Go1TerminationsCfg(TerminationsCfg):
    """Go1 termination configuration."""

    # Terminate on base contact
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    # Terminate on leg body contact (thigh, calf)
    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf",
                           "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"],
            ),
            "threshold": 1.0,
        },
    )


##
# Commands Configuration
##


@configclass
class Go1CommandsCfg(CommandsCfg):
    """Go1 command configuration."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.57, 1.57),
        ),
    )


##
# AMP Configuration
##


# Motion files - update this path to your motion data
MOTION_FILES = glob.glob("/home/wzh/amp/legged_rl_lab/data/motions/go1/*")


@configclass
class Go1AMPCfg(AMPCfg):
    """Go1 AMP-specific configuration."""

    motion_files: list[str] = MOTION_FILES
    reference_state_initialization: bool = True
    reference_state_initialization_prob: float = 0.85
    
    amp_reward_coef: float = 2.0
    amp_task_reward_lerp: float = 0.3
    amp_discr_hidden_dims: list[int] = [1024, 512]
    
    amp_replay_buffer_size: int = 1000000
    amp_num_preload_transitions: int = 200000
    
    # Go1 specific: 12 joints
    # AMP obs = joint_pos(12) + foot_pos(12) + lin_vel(3) + ang_vel(3) + joint_vel(12) + z(1) = 43
    amp_obs_dim: int = 43


##
# Main Go1 AMP Environment Configuration
##


@configclass
class Go1AMPEnvCfg(AMPEnvCfg):
    """Go1 AMP environment configuration."""

    # Scene
    scene: Go1AMPSceneCfg = Go1AMPSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP components
    observations: Go1ObservationsCfg = Go1ObservationsCfg()
    actions: Go1ActionsCfg = Go1ActionsCfg()
    commands: Go1CommandsCfg = Go1CommandsCfg()
    events: Go1EventCfg = Go1EventCfg()
    rewards: Go1RewardsCfg = Go1RewardsCfg()
    terminations: Go1TerminationsCfg = Go1TerminationsCfg()

    # AMP configuration
    amp: Go1AMPCfg = Go1AMPCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        # Go1 specific simulation settings
        self.decimation = 6
        self.sim.dt = 1.0 / 200.0
        self.episode_length_s = 20.0
        
        # Viewer
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)
