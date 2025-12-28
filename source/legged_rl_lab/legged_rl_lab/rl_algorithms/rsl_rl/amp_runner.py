"""
AMP On-Policy Runner for Isaac Lab environments.

This runner handles the training loop for AMP-PPO with Isaac Lab's ManagerBasedRLEnv.

Adapted from the original RSL_RL implementation.
"""

import time
import os
from collections import deque
import statistics

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.utils.utils import Normalizer

from legged_rl_lab.rl_algorithms.rsl_rl import AMPPPO, AMPDiscriminator
from legged_rl_lab.utils.amp_loader import AMPLoader


class AMPOnPolicyRunner:
    """
    On-policy runner for AMP training with Isaac Lab environments.
    
    Handles:
    - Rollout collection
    - AMP discriminator reward computation
    - Policy/discriminator updates
    - Logging and checkpointing
    
    Args:
        env: Isaac Lab ManagerBasedRLEnv instance
        train_cfg: Training configuration dictionary
        log_dir: Directory for logging
        device: Torch device
    """
    
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        # Determine observation dimensions
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        
        # Handle observation history
        if hasattr(self.env, 'include_history_steps') and self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs
        
        # Create policy network
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        
        # Setup AMP components
        amp_data = AMPLoader(
            device=device,
            time_between_frames=self.env.dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"]
        )
        
        amp_normalizer = Normalizer(amp_data.observation_dim)
        
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,  # (state, next_state) pair
            self.cfg['amp_reward_coef'],
            self.cfg['amp_discr_hidden_dims'],
            device,
            self.cfg['amp_task_reward_lerp']
        ).to(self.device)
        
        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"])
        
        # Compute min_std if needed
        if hasattr(self.env, 'dof_pos_limits'):
            min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]))
            )
        else:
            min_std = None
        
        self.alg: AMPPPO = alg_class(
            actor_critic,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg
        )
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # Initialize storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [self.env.num_privileged_obs] if self.env.num_privileged_obs is not None else [num_critic_obs],
            [self.env.num_actions]
        )
        
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # Reset environment
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Main training loop.
        
        Args:
            num_learning_iterations: Number of training iterations
            init_at_random_ep_len: Whether to randomize initial episode lengths
        """
        # Initialize tensorboard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        # Randomize episode lengths if requested
        if init_at_random_ep_len and hasattr(self.env, 'episode_length_buf'):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        
        # Get initial observations
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        
        # Get AMP observations (from observation manager)
        if hasattr(self.env.observation_manager, 'compute_group'):
            amp_obs = self.env.observation_manager.compute_group("amp")
        else:
            # Fallback: compute AMP obs directly
            amp_obs = self.env.get_amp_observations()
        
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        
        # Set to training mode
        self.alg.actor_critic.train()
        self.alg.discriminator.train()
        
        # Episode tracking
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # ========== Rollout ==========
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # Get actions
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    
                    # Step environment
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    
                    # Get next AMP observations
                    if hasattr(self.env.observation_manager, 'compute_group'):
                        next_amp_obs = self.env.observation_manager.compute_group("amp")
                    else:
                        next_amp_obs = self.env.get_amp_observations()
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    
                    # Handle terminal AMP states
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if hasattr(self.env, 'reset_terminated') and self.env.reset_terminated:
                        # Get terminal states for environments that reset
                        reset_env_ids = dones.nonzero(as_tuple=False).flatten()
                        if len(reset_env_ids) > 0:
                            # Terminal AMP obs are stored before reset
                            if hasattr(self.env, 'terminal_amp_states'):
                                next_amp_obs_with_term[reset_env_ids] = self.env.terminal_amp_states[reset_env_ids]
                    
                    # Compute AMP reward
                    amp_rewards, disc_logits = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer
                    )
                    
                    # Update for next step
                    amp_obs = torch.clone(next_amp_obs)
                    
                    # Store transition
                    self.alg.process_env_step(amp_rewards, dones, infos, next_amp_obs_with_term)
                    
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    
                    cur_reward_sum += amp_rewards
                    cur_episode_length += 1
                    
                    new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(new_ids) > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
                # Compute returns
                stop = time.time()
                collection_time = stop - start
                
                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # ========== Update ==========
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            
            # ========== Logging ==========
            if self.log_dir is not None:
                self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
                self.tot_time += collection_time + learn_time
                
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/value_function', mean_value_loss, it)
                self.writer.add_scalar('Loss/surrogate', mean_surrogate_loss, it)
                self.writer.add_scalar('Loss/amp', mean_amp_loss, it)
                self.writer.add_scalar('Loss/grad_penalty', mean_grad_pen_loss, it)
                self.writer.add_scalar('AMP/policy_pred', mean_policy_pred, it)
                self.writer.add_scalar('AMP/expert_pred', mean_expert_pred, it)
                self.writer.add_scalar('Policy/learning_rate', self.alg.learning_rate, it)
                
                if len(rewbuffer) > 0:
                    self.writer.add_scalar('Train/mean_reward', statistics.mean(rewbuffer), it)
                    self.writer.add_scalar('Train/mean_episode_length', statistics.mean(lenbuffer), it)
                
                self.writer.add_scalar('Perf/total_fps', self.tot_timesteps / self.tot_time, it)
                self.writer.add_scalar('Perf/collection_time', collection_time, it)
                self.writer.add_scalar('Perf/learning_time', learn_time, it)
                
                # Print progress
                if it % 10 == 0:
                    print(f"Iter {it}/{tot_iter} | "
                          f"Reward: {statistics.mean(rewbuffer):.2f} | "
                          f"FPS: {int(self.tot_timesteps / self.tot_time)}")
            
            self.current_learning_iteration += 1
            ep_infos.clear()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
        }, path)
        print(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.alg.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_learning_iteration = checkpoint['iter']
        print(f"Loaded model from {path}")
    
    def get_inference_policy(self, device=None):
        """Get policy for inference/deployment."""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
