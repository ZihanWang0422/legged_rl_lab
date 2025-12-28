# AMP Migration Guide: IsaacGym → Isaac Lab

本文档说明如何将 AMP (Adversarial Motion Priors) 从旧的 IsaacGym/legged_gym 迁移到 Isaac Lab 的 legged_rl_lab。

## 1. 目录结构

```
legged_rl_lab/tasks/locomotion/amp/
├── __init__.py                 # 模块导出和环境注册
├── amp_env_cfg.py              # 基础 AMP 环境配置
├── utils/amp_utils.py          # AMP 工具函数（运动学、观测构建）
├── mdp/
│   ├── __init__.py             # MDP 函数导出
│   ├── observations.py         # 观测函数（策略输入、AMP 判别器输入）
│   ├── rewards.py              # 奖励函数
│   ├── terminations.py         # 终止条件
│   └── curriculums.py          # 课程学习
└── config/
    ├── go1/
    │   ├── __init__.py         # Go1 环境注册
    │   ├── go1_amp_env_cfg.py  # Go1 专用配置
    │   └── agents/
    │       ├── rsl_rl_ppo_cfg.py
    │       ├── skrl_amp_cfg.yaml
    │       └── ...
    └── g1/
        └── ...                 # G1 配置（类似结构）
```

## 2. 迁移对照表

| 旧代码 (IsaacGym)                          | 新代码 (Isaac Lab)                              |
|--------------------------------------------|------------------------------------------------|
| `go1_amp_config.py`                        | `go1_amp_env_cfg.py`                           |
| `LeggedRobotCfg` class                     | `@configclass AMPEnvCfg`                       |
| `legged_robot.py` rewards                  | `mdp/rewards.py`                               |
| `legged_robot.py` observations             | `mdp/observations.py`                          |
| `check_termination()`                      | `mdp/terminations.py`                          |
| `amp_on_policy_runner.py`                  | 使用 skrl AMP agent 或自定义 runner            |
| `AMPLoader`                                | skrl motion_dataset / 自定义加载器             |

## 3. MDP 流程设计

### 3.1 状态空间 (Observations)

**策略输入 (Policy)**
- `base_ang_vel`: 基座角速度 (3)
- `projected_gravity`: 重力投影 (3)
- `velocity_commands`: 速度指令 [vx, vy, ωz] (3)
- `joint_pos_rel`: 关节位置相对默认值 (12)
- `joint_vel`: 关节速度 (12)
- `last_action`: 上一步动作 (12)
- **总计**: 45 维

**特权观测 (Critic)**
- 包含策略观测 + `base_lin_vel` (3)
- **总计**: 48 维

**AMP 观测 (Discriminator)**
- `joint_pos`: 关节位置 (12)
- `foot_pos_in_base`: 足端位置 (12)
- `base_lin_vel`: 线速度 (3)
- `base_ang_vel`: 角速度 (3)
- `joint_vel`: 关节速度 (12)
- `z_pos`: 高度 (1)
- **总计**: 43 维

### 3.2 动作空间 (Actions)

- **类型**: 关节位置目标 (PD 控制)
- **维度**: 12 (Go1: 4腿 × 3关节)
- **缩放**: `action_scale = 0.25`
- **偏移**: 使用默认关节角度

### 3.3 奖励设计 (Rewards)

```python
# 任务奖励（与 AMP 判别器奖励混合）
total_reward = lerp * task_reward + (1 - lerp) * amp_reward

# 任务奖励组成
task_reward = w1 * track_lin_vel_xy + w2 * track_ang_vel_z + penalties

# AMP 奖励（来自判别器）
amp_reward = amp_reward_coef * discriminator_output
```

**奖励项**:
- `track_lin_vel_xy_exp`: 线速度跟踪 (exp kernel)
- `track_ang_vel_z_exp`: 角速度跟踪 (exp kernel)
- `lin_vel_z_l2`: z 方向速度惩罚
- `ang_vel_xy_l2`: xy 角速度惩罚
- `joint_torques_l2`: 力矩惩罚
- `action_rate_l2`: 动作变化惩罚
- `undesired_contacts`: 非法接触惩罚

### 3.4 终止条件 (Terminations)

- `time_out`: 超时终止
- `illegal_contact`: 非法身体部位接触地面（base, thigh, calf）
- `base_height_below`: 高度过低
- `base_orientation_limit`: 姿态超限 (roll/pitch)

### 3.5 课程学习 (Curriculum)

- `velocity_command_range`: 逐步增加速度指令范围
- `push_velocity`: 逐步增加推力扰动
- `amp_task_reward_lerp`: 逐步增加任务奖励比例

## 4. 配置参数对照

### 4.1 环境参数

```python
# 旧代码
class GO1AMPCfg(LeggedRobotCfg):
    class env:
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 48

# 新代码
@configclass
class Go1AMPEnvCfg(AMPEnvCfg):
    scene = Go1AMPSceneCfg(num_envs=4096, env_spacing=2.5)
    # 观测维度自动从 ObservationsCfg 推断
```

### 4.2 控制参数

```python
# 旧代码
class control:
    control_type = 'P'
    stiffness = {'joint': 80.}
    damping = {'joint': 1.0}
    action_scale = 0.25
    decimation = 6

# 新代码
actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        stiffness=80.0,
        damping=1.0,
    ),
}
# decimation 在 __post_init__ 中设置
```

### 4.3 AMP 参数

```python
@configclass
class Go1AMPCfg(AMPCfg):
    motion_files = glob.glob("datasets/mocap_motions/*")
    reference_state_initialization = True
    reference_state_initialization_prob = 0.85
    amp_reward_coef = 2.0
    amp_task_reward_lerp = 0.3
    amp_discr_hidden_dims = [1024, 512]
    amp_replay_buffer_size = 1000000
```

## 5. 使用方法

### 5.1 训练

```bash
# 使用 skrl
python train.py --task Go1-AMP-v0 --agent skrl_amp

# 使用 rsl_rl (需要自定义 AMP runner)
python train.py --task Go1-AMP-v0 --agent rsl_rl
```

### 5.2 评估

```bash
python play.py --task Go1-AMP-v0 --checkpoint /path/to/model.pt
```

## 6. TODO / 待实现

1. **足端运动学**: 在 `observations.py` 中实现 `_compute_foot_positions_in_base_frame()`
2. **运动数据加载**: 集成 motion loader 到环境或 runner
3. **参考状态初始化**: 在 reset 事件中实现 AMP 参考状态初始化
4. **自定义 AMP Runner**: 如果使用 rsl_rl，需要移植 `amp_on_policy_runner.py`
5. **G1 配置**: 创建 G1 机器人的 AMP 配置

## 7. 注意事项

- 确保 URDF/USD 文件路径正确
- 运动数据文件格式需要与 loader 兼容
- 调整观测噪声以匹配训练需求
- 根据实际机器人调整运动学参数
