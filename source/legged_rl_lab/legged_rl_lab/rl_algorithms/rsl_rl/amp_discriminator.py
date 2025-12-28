"""
AMP (Adversarial Motion Priors) Discriminator for Isaac Lab.

Adapted from the original RSL_RL implementation for use with Isaac Lab environments.
"""

import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd


class AMPDiscriminator(nn.Module):
    """
    Discriminator network for Adversarial Motion Priors (AMP).
    
    The discriminator distinguishes between:
    - Expert demonstrations (from motion capture data)
    - Agent-generated trajectories
    
    Args:
        input_dim: Dimension of state-transition pairs (s_t, s_{t+1})
        amp_reward_coef: Coefficient for AMP reward scaling
        hidden_layer_sizes: List of hidden layer dimensions
        device: Torch device (cuda/cpu)
        task_reward_lerp: Linear interpolation weight between AMP and task rewards (0-1)
    """
    
    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: list[int],
        device: str,
        task_reward_lerp: float = 0.0
    ):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        # Build discriminator trunk (feature extractor)
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        # Set to training mode
        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x: Input state-transition pairs [batch, input_dim]
            
        Returns:
            Discriminator logits [batch, 1]
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP style training.
        
        Enforces Lipschitz constraint by penalizing gradient norm deviation from 0.
        
        Args:
            expert_state: Expert state at time t
            expert_next_state: Expert state at time t+1  
            lambda_: Gradient penalty coefficient
            
        Returns:
            Gradient penalty loss
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Penalize deviation from zero gradient norm
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict AMP-style reward from discriminator.
        
        Uses the discriminator output to compute a reward that encourages
        the agent to match expert behavior.
        
        Args:
            state: Current state
            next_state: Next state
            task_reward: Task-specific reward (for lerp)
            normalizer: Optional state normalizer
            
        Returns:
            - AMP reward [batch]
            - Discriminator logits [batch, 1]
        """
        with torch.no_grad():
            self.eval()
            
            # Normalize states if normalizer provided
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            # Get discriminator output
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            
            # Compute AMP reward: r = coef * max(0, 1 - 0.25*(d-1)^2)
            reward = self.amp_reward_coef * torch.clamp(
                1 - (1/4) * torch.square(d - 1), min=0
            )
            
            # Optionally blend with task reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            
            self.train()
            
        return reward.squeeze(), d

    def _lerp_reward(
        self,
        disc_r: torch.Tensor,
        task_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation between discriminator and task rewards.
        
        Args:
            disc_r: Discriminator reward
            task_r: Task reward
            
        Returns:
            Blended reward
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r

    def save(self, path: str):
        """Save discriminator weights."""
        torch.save({
            'trunk_state_dict': self.trunk.state_dict(),
            'amp_linear_state_dict': self.amp_linear.state_dict(),
        }, path)

    def load(self, path: str):
        """Load discriminator weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.trunk.load_state_dict(checkpoint['trunk_state_dict'])
        self.amp_linear.load_state_dict(checkpoint['amp_linear_state_dict'])
