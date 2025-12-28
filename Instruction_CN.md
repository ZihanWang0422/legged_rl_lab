# Velocity Locomotion

## MDP 分层导入结构
```python
# mdp/__init__.py
from isaaclab.envs.mdp import *                           # 1. 基础 MDP 函数
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # 2. 官方 velocity MDP
from .curriculums import *                                # 3. 自定义课程学习
from .rewards import *                                    # 4. 自定义奖励
from .terminations import *                               # 5. 自定义终止条件
from .commands import *                                   # 6. 自定义命令
from .observations import *                               # 7. 自定义观察
```


## RL Library

- [isaaclab.rl](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)

- [RL Scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)

- [Add own rl library](https://isaac-sim.github.io/IsaacLab/main/source/how-to/add_own_library.html)

- [Config rl agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/configuring_rl_training.html)

- [Train rl agent](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html)

## Velocity_env_cfg.py

- [isaaclab.scene](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg)

- [isaaclab.managers](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html)

其继承了manager env中的termcfg
- [isaaclab.envs.mdp](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)


- [isaaclab.envs.ManagerBasedRLEnv](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv)
- [综合定义](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)