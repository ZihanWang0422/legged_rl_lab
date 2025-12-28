#!/usr/bin/env python3
"""Play script for trained AMP locomotion policies.

This script loads a trained AMP policy and visualizes it in Isaac Lab.

Example usage:
    python play_amp.py --task Go1-Amp-v0 --checkpoint logs/amp_locomotion/run_name/model_1000.pt
    python play_amp.py --task Go1-Amp-v0 --load_run run_name
"""

import argparse
import os

import torch
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play trained AMP locomotion policy")
parser.add_argument("--task", type=str, default="Go1-Amp-v0", help="Name of the task")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (default: task default)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of run to load")
parser.add_argument("--experiment_name", type=str, default="amp_locomotion", help="Experiment name")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint in run")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim (never headless for play script)
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from legged_rl_lab.rl_algorithms.rsl_rl import AMPPPO


def main():
    """Main play function."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 1,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Modify environment for visualization
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.viewer.eye = (5.0, 5.0, 3.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Determine checkpoint path
    checkpoint_path = None
    if args_cli.checkpoint:
        checkpoint_path = args_cli.checkpoint
    elif args_cli.load_run:
        log_root = os.path.join("logs", args_cli.experiment_name, args_cli.load_run)
        if args_cli.use_last_checkpoint or not os.path.exists(os.path.join(log_root, "model.pt")):
            # Find last checkpoint
            checkpoints = [f for f in os.listdir(log_root) if f.startswith("model_") and f.endswith(".pt")]
            if checkpoints:
                checkpoint_nums = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
                last_checkpoint = checkpoints[checkpoint_nums.index(max(checkpoint_nums))]
                checkpoint_path = os.path.join(log_root, last_checkpoint)
            else:
                print(f"[ERROR] No checkpoints found in {log_root}")
                return
        else:
            checkpoint_path = os.path.join(log_root, "model.pt")
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    # Import configuration based on task
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
    
    # Import policy class
    from rsl_rl.modules import ActorCritic
    
    policy = ActorCritic(
        num_actor_obs=env.observation_space["policy"].shape[0],
        num_critic_obs=env.observation_space["policy"].shape[0],
        num_actions=env.action_space.shape[0],
        init_noise_std=agent_cfg.policy.init_noise_std,
        actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
        activation=agent_cfg.policy.activation,
    ).to(env.device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    
    print("\n" + "=" * 80)
    print(f"Playing AMP Policy: {args_cli.task}")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Number of environments: {env.num_envs}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print("=" * 80 + "\n")
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Simulation loop
    with torch.no_grad():
        while simulation_app.is_running():
            # Get actions from policy
            actions = policy.act_inference(obs)
            
            # Step environment
            obs_dict, rewards, dones, truncated, infos = env.step(actions)
            obs = obs_dict["policy"]
            
            # Print reward info occasionally
            if env.episode_length_buf[0] % 100 == 0:
                print(f"Episode length: {env.episode_length_buf[0].item()}, Mean reward: {rewards.mean().item():.3f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Play failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()
