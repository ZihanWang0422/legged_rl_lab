#!/usr/bin/env python3
"""Training script for AMP locomotion using RSL-RL.

This script demonstrates how to train a quadruped robot using Adversarial Motion Priors (AMP)
with the RSL-RL algorithm implementation.

Example usage:
    python train_amp.py --task Go1-Amp-v0 --headless --num_envs 4096
    python train_amp.py --task Go1-Amp-v0 --resume --load_run run_name
"""

import argparse
import os
from datetime import datetime

import torch
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train AMP locomotion policy with RSL-RL")
parser.add_argument("--task", type=str, default="Go1-Amp-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")

# Training arguments
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Steps per environment per rollout")
parser.add_argument("--save_interval", type=int, default=50, help="Model save interval (iterations)")

# Resume training
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of run to resume")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

# AMP-specific arguments
parser.add_argument("--amp_motion_files", type=str, default=None, 
                    help="Comma-separated list of motion capture files")
parser.add_argument("--amp_reward_coef", type=float, default=2.0,
                    help="Coefficient for AMP discriminator reward")
parser.add_argument("--amp_task_reward_lerp", type=float, default=0.3,
                    help="Task reward interpolation (0=only AMP, 1=only task)")

# Logging
parser.add_argument("--logger", type=str, default="tensorboard",
                    choices=["tensorboard", "neptune", "wandb"],
                    help="Logger backend")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
parser.add_argument("--run_name", type=str, default="", help="Run name (default: auto-generated)")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from legged_rl_lab.rl_algorithms.rsl_rl import AMPOnPolicyRunner


def main():
    """Main training function."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Override seed if provided
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Import agent configuration based on task
    # For Go1 tasks, use Go1-specific config
    if "go1" in args_cli.task.lower():
        if "flat" in args_cli.task.lower():
            from legged_rl_lab.tasks.locomotion.amp.config.go1.agents.rsl_rl_ppo_cfg import Go1AMPRunnerFlatCfg
            agent_cfg = Go1AMPRunnerFlatCfg()
        else:
            from legged_rl_lab.tasks.locomotion.amp.config.go1.agents.rsl_rl_ppo_cfg import Go1AMPRunnerRoughCfg
            agent_cfg = Go1AMPRunnerRoughCfg()
    else:
        # Fallback to generic Go1 config
        from legged_rl_lab.tasks.locomotion.amp.config.go1.agents.rsl_rl_ppo_cfg import Go1AMPRunnerCfg
        agent_cfg = Go1AMPRunnerCfg()
    
    # Override configuration with CLI arguments
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger:
        agent_cfg.logger = args_cli.logger
    
    # AMP-specific overrides
    if args_cli.amp_motion_files is not None:
        agent_cfg.amp_motion_files = args_cli.amp_motion_files.split(",")
    if args_cli.amp_reward_coef is not None:
        agent_cfg.discriminator.amp_reward_coef = args_cli.amp_reward_coef
    if args_cli.amp_task_reward_lerp is not None:
        agent_cfg.discriminator.task_reward_lerp = args_cli.amp_task_reward_lerp
    
    # Set motion files from environment config if available
    if hasattr(env_cfg, "amp_motion_files") and agent_cfg.amp_motion_files is None:
        agent_cfg.amp_motion_files = env_cfg.amp_motion_files
    
    # Create runner
    runner = AMPOnPolicyRunner(env, agent_cfg, log_dir=None, device=env.device)
    
    # Resume training if requested
    if args_cli.resume:
        if args_cli.checkpoint:
            print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
            runner.load(args_cli.checkpoint)
        elif args_cli.load_run:
            log_root_path = os.path.join("logs", agent_cfg.experiment_name)
            resume_path = os.path.join(log_root_path, args_cli.load_run)
            print(f"[INFO] Resuming from run: {resume_path}")
            runner.load(resume_path)
        else:
            print("[ERROR] Resume flag set but no checkpoint or run name provided!")
            return
    
    # Print training info
    print("\n" + "=" * 80)
    print(f"Training AMP Policy: {args_cli.task}")
    print("=" * 80)
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Environment device: {env.device}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max iterations: {agent_cfg.max_iterations}")
    print(f"  Steps per environment: {agent_cfg.num_steps_per_env}")
    print(f"  AMP motion files: {agent_cfg.amp_motion_files}")
    print(f"  AMP reward coefficient: {agent_cfg.discriminator.amp_reward_coef}")
    print(f"  Task/AMP reward lerp: {agent_cfg.discriminator.task_reward_lerp}")
    print("=" * 80 + "\n")
    
    # Start training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()
