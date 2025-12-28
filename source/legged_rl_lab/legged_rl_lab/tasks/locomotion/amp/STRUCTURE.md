# AMP 任务目录结构说明

本目录包含了 AMP (Adversarial Motion Priors) 相关的所有代码，采用模块化设计。

## 目录结构

```
amp/
├── README.md                   # AMP 任务说明
├── STRUCTURE.md               # 本文件：目录结构说明
├── __init__.py                # 包初始化
├── amp_env_cfg.py            # 主环境配置文件
│
├── managers/                  # 数据和动画管理器
│   ├── __init__.py
│   ├── motion_data_manager.py      # 运动数据加载和管理
│   ├── motion_data_term_cfg.py     # 运动数据配置
│   ├── animation_manager.py        # 动画播放和插值
│   └── animation_manager_cfg.py    # 动画配置
│
├── envs/                      # 环境基类
│   ├── __init__.py
│   ├── manager_based_animation_env.py      # 动画环境基类
│   ├── manager_based_animation_env_cfg.py  # 动画环境配置
│   ├── manager_based_amp_env.py           # AMP 环境实现
│   └── manager_based_amp_env_cfg.py       # AMP 环境配置
│
├── utils/                     # 工具函数
│   ├── __init__.py
│   └── math.py               # 数学工具（速度计算、四元数插值等）
│
├── mdp/                       # MDP 组件（观测、奖励、事件等）
│   ├── __init__.py
│   ├── actions.py
│   ├── commands.py
│   ├── observations.py
│   ├── rewards.py
│   ├── events.py
│   └── terminations.py
│
└── config/                    # 具体机器人配置
    ├── go1/                  # GO1 机器人
    │   ├── __init__.py
    │   ├── go1_amp_env_cfg.py
    │   └── agents/
    │       └── rsl_rl_ppo_cfg.py
    └── g1/                   # G1 人形机器人
        ├── __init__.py
        ├── g1_amp_env_cfg.py
        └── agents/
            └── rsl_rl_ppo_cfg.py
```

## 模块说明

### 1. managers/ - 核心功能管理器

#### motion_data_manager.py
- **MotionDataTerm**: 加载和管理单个运动数据源
- **MotionDataManager**: 管理多个运动数据源
- 功能：
  - 从 .pkl 文件加载运动捕捉数据
  - 支持循环/截断模式
  - 数据插值和采样
  - 多种运动数据组件（位置、速度、关节角度等）

#### animation_manager.py
- **AnimationTerm**: 单个动画序列
- **AnimationManager**: 管理多个动画
- 功能：
  - 从运动数据中提取指定时间步的数据
  - 支持前向/后向历史数据
  - 随机初始化和随机采样
  - 可视化支持

### 2. envs/ - 环境类

#### manager_based_animation_env.py
- 基础动画环境，支持运动数据加载和动画播放
- 包含 `motion_data_manager` 和 `animation_manager`

#### manager_based_amp_env.py
- 继承自动画环境
- 重写 `step()` 方法以保留重置前的观测（用于判别器训练）
- AMP 特定的逻辑

### 3. utils/ - 工具函数

#### math.py
数学工具函数：
- `vel_forward_diff()`: 前向差分计算速度
- `ang_vel_from_quat_diff()`: 从四元数差分计算角速度
- `quat_slerp()`: 四元数球面线性插值
- `linear_interpolate()`: 线性插值
- `calc_frame_blend()`: 计算帧混合参数

### 4. mdp/ - MDP 组件

包含环境的观测、动作、奖励、事件和终止条件的具体实现。

### 5. config/ - 机器人配置

包含具体机器人（如 GO1、G1）的完整环境配置和训练算法配置。

## 使用方式

### 创建新的 AMP 任务

1. 在 `config/` 下创建新机器人文件夹
2. 创建环境配置文件，继承自 `AMPEnvCfg`
3. 配置 `motion_data` 和 `animation` 参数
4. 在 `agents/` 下创建训练算法配置

### 导入示例

```python
# 导入基础配置
from legged_rl_lab.tasks.locomotion.amp.amp_env_cfg import AMPEnvCfg

# 导入管理器
from legged_rl_lab.tasks.locomotion.amp.managers import (
    MotionDataManager,
    AnimationManager,
    MotionDataTermCfg,
    AnimationTermCfg,
)

# 导入环境
from legged_rl_lab.tasks.locomotion.amp.envs import (
    ManagerBasedAmpEnv,
    ManagerBasedAmpEnvCfg,
)

# 导入工具函数
from legged_rl_lab.tasks.locomotion.amp.utils import (
    vel_forward_diff,
    quat_slerp,
)
```

## 设计原则

1. **职责分离**: 配置、功能实现、环境逻辑分离
2. **可复用性**: 管理器可被多个任务复用
3. **模块化**: 每个模块独立，便于测试和维护
4. **相对导入**: 使用相对导入保持模块独立性

## 与原始项目的对比

原始 `isaacgym/AMP_for_hardware` 项目的代码分散在：
- `legged_gym/envs/base/legged_robot.py` - 环境实现
- `rsl_rl/datasets/motion_loader.py` - 数据加载
- `legged_gym/envs/go1/go1_amp_config.py` - 配置

现在全部整合到 `amp/` 目录下，结构更清晰，更易维护。
