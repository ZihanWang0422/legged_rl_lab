from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_animation_env_cfg import ManagerBasedAnimationEnvCfg

@configclass
class ManagerBasedAmpEnvCfg(ManagerBasedAnimationEnvCfg):
    """Configuration for a AMP environment with the manager-based workflow."""
    
    pass