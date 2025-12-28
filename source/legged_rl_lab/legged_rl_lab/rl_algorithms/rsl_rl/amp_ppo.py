"""
AMP-PPO (Adversarial Motion Priors with Proximal Policy Optimization) for Isaac Lab.

This module extends standard PPO with AMP discriminator training for imitation learning
from motion capture data.

Adapted from the original RSL_RL implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class AMPPPO:
    """
    AMP-PPO algorithm implementation.
    
    Combines:
    - PPO (Proximal Policy Optimization) for policy learning
    - AMP (Adversarial Motion Priors) discriminator for style reward
    
    Args:
        actor_critic: Policy network
        discriminator: AMP discriminator network
        amp_data: Motion capture data loader
        amp_normalizer: State normalizer for AMP observations
        num_learning_epochs: Number of epochs per update
        num_mini_batches: Number of mini-batches per epoch
        clip_param: PPO clipping parameter
        gamma: Discount factor
        lam: GAE lambda
        value_loss_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        use_clipped_value_loss: Whether to use clipped value loss
        schedule: Learning rate schedule ('fixed' or 'adaptive')
        desired_kl: Target KL divergence for adaptive LR
        device: Torch device
        amp_replay_buffer_size: Size of AMP replay buffer
        min_std: Minimum standard deviation for policy
    """
    
    actor_critic: ActorCritic
    
    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_data,
        amp_normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device='cpu',
        amp_replay_buffer_size=100000,
        min_std=None,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # Initialized later

        # Optimizer for policy and discriminator
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {
                'params': self.discriminator.trunk.parameters(),
                'weight_decay': 10e-4,
                'name': 'amp_trunk'
            },
            {
                'params': self.discriminator.amp_linear.parameters(),
                'weight_decay': 10e-2,
                'name': 'amp_head'
            }
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list,
        critic_obs_shape: list,
        action_shape: list
    ):
        """Initialize rollout storage."""
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device
        )

    def test_mode(self):
        """Set networks to evaluation mode."""
        self.actor_critic.test()

    def train_mode(self):
        """Set networks to training mode."""
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, amp_obs: torch.Tensor):
        """
        Compute actions and values for the current observations.
        
        Args:
            obs: Policy observations
            critic_obs: Critic observations (may include privileged info)
            amp_obs: AMP discriminator observations
            
        Returns:
            Actions to execute
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute actions and values
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Record observations before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        amp_obs: torch.Tensor
    ):
        """
        Process environment step and store transition.
        
        Args:
            rewards: Task rewards
            dones: Done flags
            infos: Environment info dict (may contain 'time_outs')
            amp_obs: Current AMP observations
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        # Store AMP transitions (for discriminator training)
        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # Record transition in storage
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        """Compute returns using GAE."""
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """
        Update policy, value function, and discriminator.
        
        Returns:
            Tuple of (value_loss, surrogate_loss, amp_loss, grad_pen_loss, 
                     policy_pred, expert_pred)
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        
        # Setup generators
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        batch_size = (
            self.storage.num_envs * self.storage.num_transitions_per_env //
            self.num_mini_batches
        )
        
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, batch_size
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches, batch_size
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            (
                obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, old_actions_log_prob_batch,
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch
            ) = sample
            
            # Policy evaluation
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(
                aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(
                aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive learning rate (KL-based)
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # AMP discriminator loss
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            
            # Discriminator wants to classify expert as +1, policy as -1
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
            # Gradient penalty for WGAN-style training
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=10
            )

            # Total loss
            loss = (
                surrogate_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy_batch.mean() +
                amp_loss + grad_pen_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Clamp std if specified
            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            # Update AMP normalizer
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # Accumulate stats
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        # Average stats
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred
        )
