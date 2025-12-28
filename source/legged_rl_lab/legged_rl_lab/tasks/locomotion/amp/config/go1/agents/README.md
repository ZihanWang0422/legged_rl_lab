# Go1 AMP Agent Configurations

This directory contains training configurations for Go1 robot using different RL frameworks.

## Available Configurations

### RSL-RL (Recommended for AMP)
- **File**: `rsl_rl_ppo_cfg.py`
- **Algorithm**: AMP-PPO (Adversarial Motion Priors with Proximal Policy Optimization)
- **Features**:
  - Custom discriminator network for motion imitation
  - Replay buffer for expert demonstrations
  - Mixed task and style rewards
  - Go1-specific hyperparameters

**Usage**:
```python
from legged_rl_lab.tasks.locomotion.amp.config.go1.agents.rsl_rl_ppo_cfg import get_go1_amp_config

agent_cfg = get_go1_amp_config()
```

### Other Frameworks
- `rl_games_ppo_cfg.yaml` - RL Games PPO
- `sb3_ppo_cfg.yaml` - Stable-Baselines3 PPO
- `skrl_ppo_cfg.yaml` - skrl PPO
- `skrl_amp_cfg.yaml` - skrl AMP

## Configuration Structure

The AMP configuration includes:

1. **Policy Network**: Actor-Critic architecture with customizable hidden layers
2. **Discriminator Network**: Distinguishes between expert and policy trajectories
3. **PPO Algorithm**: Standard PPO hyperparameters (clip, entropy, learning rate, etc.)
4. **AMP Parameters**:
   - `amp_reward_coef`: Weight for discriminator reward
   - `amp_task_reward_lerp`: Balance between task and style rewards
   - `amp_replay_buffer_size`: Size of expert demonstration buffer
   - `amp_num_preload_transitions`: Number of expert transitions to preload
5. **Runner Configuration**: Training loop settings (iterations, save intervals, logging)
6. **Go1-Specific Settings**: Robot height, joint limits, normalization parameters

## Key Parameters

### AMP Reward Balancing
```python
"amp_task_reward_lerp": 0.3  # 0.3 * task_reward + 0.7 * amp_reward
"amp_reward_coef": 2.0        # Scale factor for discriminator reward
```

### Network Architecture
```python
# Policy
"actor_hidden_dims": [512, 256, 128]
"critic_hidden_dims": [512, 256, 128]

# Discriminator
"hidden_dims": [1024, 512]
```

### Go1-Specific
```python
"base_height_target": 0.28  # Go1's nominal height
"min_normalized_std": [0.05, 0.02, 0.05] * 4  # Per joint group (hip, thigh, calf) × 4 legs
```

## Training Scripts

Use the training scripts in `/scripts/rsl_rl/`:

```bash
# Train Go1 with AMP
python scripts/rsl_rl/train_amp.py \
    --task Go1-Amp-v0 \
    --headless \
    --num_envs 4096

# Visualize trained policy
python scripts/rsl_rl/play_amp.py \
    --task Go1-Amp-v0 \
    --load_run <run_name>
```

## Customization

To create custom configurations:

1. Copy `rsl_rl_ppo_cfg.py` to a new file
2. Modify the parameters in `get_go1_amp_config()`
3. Import and use in your training script:
   ```python
   from your_module import get_custom_config
   agent_cfg = get_custom_config()
   ```

## References

- Original AMP Paper: [Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180)
- RSL-RL: [Robotics Systems Lab Reinforcement Learning Library](https://github.com/leggedrobotics/rsl_rl)
- IsaacGym AMP: [AMP for Hardware Project](https://github.com/ZihanWang0422/AMP_for_hardware)
