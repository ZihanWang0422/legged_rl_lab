"""
RSL_RL algorithm extensions for AMP and other advanced methods.

This module extends the standard RSL_RL library with:
- AMP (Adversarial Motion Priors) integration
- Custom discriminators
- Specialized runners
"""

from .amp_discriminator import AMPDiscriminator
from .amp_ppo import AMPPPO
from .amp_runner import AMPOnPolicyRunner

__all__ = [
    "AMPDiscriminator",
    "AMPPPO", 
    "AMPOnPolicyRunner",
]
