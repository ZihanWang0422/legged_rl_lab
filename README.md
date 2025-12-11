# legged_rl_lab

## Overview



## Installation

- Install Isaac Lab:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html


- Clone `legged_rl_lab` separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/legged_rl_lab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.


## Project Structure
https://github.com/isaac-sim/IsaacLabExtensionTemplate/tree/main/source/ext_template/ext_template/tasks/locomotion/velocity/config







## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/legged_rl_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```







## Acknowledgements

### rl_locomotion

* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [basic-locomotion-dls-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-dls-isaaclab)
* [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab?tab=readme-ov-file#acknowledgements)
* [LeggedLab](https://github.com/Hellod035/LeggedLab)
* [parkour_lab](https://github.com/CAI23sbP/Isaaclab_Parkour)
* [wheel_legged_lab](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot)

### AMP/IL_locomotion

* [legged_lab](https://github.com/zitongbai/legged_lab)
* [MimicKit](https://github.com/xbpeng/MimicKit)
* [beyondAMP](https://github.com/Renforce-Dynamics/beyondAMP)
* [motion_imitation](https://github.com/erwincoumans/motion_imitation/tree/master)


### motion_tracking_WBC

* [holosoma](https://github.com/amazon-far/holosoma?tab=readme-ov-file)

### loco_mani_WBC

### navigation

* [isaac-go2-ros2](https://github.com/Zhefan-Xu/isaac-go2-ros2)
* [legged-loco](https://github.com/yang-zj1026/legged-loco)
* [go2-ros2](https://github.com/abizovnuralem/go2_omniverse)

### mujoco

* [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
* [mjlab](https://github.com/mujocolab/mjlab)
* [mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
* [FastTD3](https://github.com/younggyoseo/FastTD3)