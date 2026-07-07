# G1 Python Deployment

This directory contains the Python deployment scripts for Unitree G1 29-DoF policies.
It covers MuJoCo Sim2Sim validation, local SDK2 closed-loop simulation, and Sim2Real
deployment on the robot.

Unless noted otherwise, run the commands below from:

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
```

For the Chinese version, see [README_CN.md](README_CN.md).

## Sim2Sim With MuJoCo

### 1. SDK2 Environment Setup

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy

# MuJoCo / ONNX / gamepad dependencies
pip install onnxruntime scipy pyyaml mujoco pygame

# CycloneDDS C library. Skip this block if deploy/g1_deploy/cyclonedds/install already exists.
python -m pip install cmake
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
python -m cmake .. -DCMAKE_INSTALL_PREFIX=../install
python -m cmake --build . --target install -j"$(nproc)"

# unitree_sdk2_python submodule
cd ~/legged_rl_lab
git submodule update --init --recursive deploy/g1_deploy/g1_python/unitree_sdk2_python

cd deploy/g1_deploy/g1_python/unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../../cyclonedds/install
pip install -e .

# Keep IsaacLab-compatible dependency versions if pip upgraded them.
pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

Verify the installation:

```bash
python -c "import cyclonedds, unitree_sdk2py, mujoco, pygame; print('sdk2 sim env ok')"
```

Local simulation uses DDS domain `1`; the real robot normally uses domain `0`.
The message `selected interface "lo" is not multicast-capable: disabling multicast`
is normal on loopback and is not an error.

### 2. SDK2 Closed-Loop Sim2Sim Flow

The local SDK2 loop uses two terminals:

- Terminal 1 runs MuJoCo and the SDK2 bridge.
- Terminal 2 runs the deploy controller, using the same style of entry point as Sim2Real.

Data flow:

```text
sim2real_*.py publishes rt/lowcmd
  -> sim2sim_sdk2_bridge.py receives LowCmd and drives MuJoCo actuators
  -> MuJoCo steps the simulated robot
  -> bridge publishes rt/lowstate, rt/wirelesscontroller, and rt/sportmodestate
  -> sim2real_*.py reads LowState, runs policy inference, and publishes the next LowCmd
```

#### 2.1 Optional: Plain Sim2Sim First

This path does not use DDS. Use it first to verify the ONNX model, YAML config,
gamepad mapping, and policy behavior.

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

Keyboard input:

```bash
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input keyboard
```

#### 2.2 Terminal 1: Start The SDK2 Bridge

Gamepad input:

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

Keyboard input:

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input keyboard \
  --elastic_band \
  --debug_lowcmd
```

Keep the MuJoCo viewer open. If the bridge prints `cmd_age=inf`, no controller has
published `rt/lowcmd` yet; this is expected before Terminal 2 starts.

`--debug_lowcmd` prints the LowCmd-to-torque range. The bridge clamps torque to the
MuJoCo XML `ctrlrange` by default. If `ctrl_raw` is very large or `clipped` stays
nonzero, check observation alignment, action scale, default joint pose, and PD gains
before suspecting DDS timing.

#### 2.3 Terminal 2: Start The Walk Controller

```bash
conda activate legged_rl_lab
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
python sim2real_walk.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_walk.yaml \
  --debug_policy
```

Both terminals must use the same `--net` and `--domain_id`. For local simulation,
use `--net lo --domain_id 1`. If CycloneDDS does not use `lo`, set both terminals
to the same interface, such as `enp108s0`.

`[PolicyDebug]` prints policy input and output ranges. Near a stable standing pose,
`grav` should be close to `[0, 0, -1]` and `cmd` should be close to zero.

#### 2.4 Operating Sequence

1. Start Terminal 1 and keep the MuJoCo viewer open.
2. Start Terminal 2.
3. For keyboard input, focus the Terminal 1 bridge window before pressing keys.
4. When the controller prints `Waiting for the start signal to move to default pos...`, press gamepad **Start/+** or keyboard **Enter / 1**.
5. Wait for the robot to move to the default pose.
6. When the controller prints `Waiting for the Button A signal to Start Control...`, press gamepad **A** or keyboard **2**.
7. Keep virtual support enabled at first and watch `[PolicyDebug]` and `[LowCmdDebug]`.
8. Lower the virtual support gradually in the MuJoCo viewer.
9. Press gamepad **Select/-** or keyboard **9 / Esc** to exit.

#### 2.5 Gamepad And Keyboard Controls

| Input | Function |
| --- | --- |
| **Start/+** or **Enter / 1** | Move from zero torque to default pose |
| **A** or **2** | Start policy control |
| Left stick up/down or **W/S** | Forward velocity command |
| Left stick left/right or **A/D** | Lateral velocity command |
| Right stick left/right or **Q/E** | Yaw-rate command |
| **Space / 0** | Clear velocity commands |
| **Select/-** or **9 / Esc** | Exit / damping |

`Select` is the exit key. Use **Start/+** to enter the default-pose stage.

#### 2.6 Virtual Support Band

The virtual support keys only work when the MuJoCo viewer is focused.

| MuJoCo Viewer Input | Function |
| --- | --- |
| **9** | Toggle support |
| **8** | Increase support length and lower the robot |
| **7** | Decrease support length and lift the robot |

Do not disable support immediately at startup. Lower it gradually after the controller
is publishing stable LowCmd values.

## Policy Command Reference

The commands below list plain Sim2Sim and local SDK2 closed-loop runs for each policy.
Real-robot Sim2Real commands are listed in the Sim2Real section.

Available exported policies:

| ONNX | Purpose | Script | Input | Output |
| --- | --- | --- | --- | --- |
| `g1_flat_1.onnx` | Flat walking, stable standing, velocity control | `sim2sim_walk.py`; startup policy for `sim2sim_mimic.py` | `obs [1, 96]` | `actions [1, 29]` |
| `g1_walk.onnx` | AMP walking policy | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_run.onnx` | AMP running policy | `sim2sim_amp.py` | `obs [1, 384]` | `actions [1, 29]` |
| `g1_dance.onnx` | Mimic dance policy | `sim2sim_mimic.py` | `obs [1, 160]`, `time_step [1, 1]` | `actions [1, 29]` and reference state |
| `g1_jump.onnx` | Mimic jump policy | `sim2sim_mimic.py` | `obs [1, 160]`, `time_step [1, 1]` | `actions [1, 29]` and reference state |
| `g1_attention1.onnx` / `g1_attention2.onnx` | Attention terrain / parkour policy | `sim2sim_attention.py` | `obs [1, 2175]` | `actions [1, 29]` |

### Walk

Plain Sim2Sim:

```bash
python sim2sim_walk.py \
  --config g1_walk.yaml \
  --model g1_flat_1.onnx \
  --input gamepad
```

SDK2 Terminal 1:

```bash
python sim2sim_sdk2_bridge.py \
  --config g1_walk.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 Terminal 2:

```bash
python sim2real_walk.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_walk.yaml \
  --debug_policy
```

### AMP Walk / Run

Plain Sim2Sim:

```bash
python sim2sim_amp.py \
  --config g1_amp.yaml \
  --model g1_walk.onnx \
  --input gamepad
```

SDK2 Terminal 1:

```bash
python sim2sim_sdk2_bridge.py \
  --config g1_amp.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 Terminal 2:

```bash
python sim2real_amp.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx \
  --debug_policy
```

For running, replace `--model g1_walk.onnx` with `--model g1_run.onnx` in both places.

### Mimic Dance / Jump

Plain Sim2Sim:

```bash
python sim2sim_mimic.py \
  --config g1_mimic.yaml \
  --model g1_dance.onnx \
  --input gamepad
```

SDK2 Terminal 1:

```bash
python sim2sim_sdk2_bridge.py \
  --config g1_mimic.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 Terminal 2:

```bash
python sim2real_mimic.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx
```

For jumping, replace `g1_dance.onnx` with `g1_jump.onnx`. `sim2sim_mimic.py`
first uses `g1_flat_1.onnx` for stable standing, then switches to the tracking
policy after the robot is stable and gamepad **B** is pressed.

### Attention Terrain / Parkour

If `../exported_policy/` does not contain `g1_attention1.onnx` or `g1_attention2.onnx`,
export it from the training result and place it under `deploy/g1_deploy/exported_policy/`,
or pass an absolute path with `--model /path/to/your_attention.onnx`.

Plain Sim2Sim with gamepad input:

```bash
python sim2sim_attention.py \
  --config g1_attention.yaml \
  --model g1_attention2.onnx \
  --input gamepad \
  --gamepad_type gamesir
```

Keyboard input:

```bash
python sim2sim_attention.py \
  --config g1_attention.yaml \
  --model g1_attention2.onnx \
  --input keyboard
```

Validate config and ONNX dimensions without opening the viewer:

```bash
python sim2sim_attention.py \
  --config g1_attention.yaml \
  --model g1_attention2.onnx \
  --check
```

Debug policy with per-joint prints:

```bash
python sim2sim_attention.py \
  --config g1_attention.yaml \
  --model g1_attention2.onnx \
  --input gamepad \
  --debug_policy \
  --debug_interval 1.0
```

SDK2 Terminal 1:

```bash
python sim2sim_sdk2_bridge.py \
  --config g1_attention.yaml \
  --net lo \
  --domain_id 1 \
  --input gamepad \
  --joystick_type switch \
  --elastic_band \
  --debug_lowcmd
```

SDK2 Terminal 2:

```bash
python sim2real_attention.py \
  --net lo \
  --domain_id 1 \
  --config_path config/g1_attention.yaml \
  --model g1_attention2.onnx \
  --debug_policy
```

The attention policy uses MuJoCo `mj_ray` in plain Sim2Sim mode to scan terrain
around the robot. In SDK2 closed-loop mode, the bridge runs MuJoCo and the terrain
scan happens inside the bridge (the bridge publishes `LowState`; terrain scan is
handled on Terminal 2 side with a flat placeholder map). For the real robot, keep
`--terrain_source flat` (the default) since there is no physical terrain scanner.

The attention policy has `history_length=1` and takes a single-frame observation of
`proprio(96) + terrain_map(2079) = 2175` dimensions. The terrain map is a 33×21 grid
of `[local_x, local_y, local_z]` points in the sensor-body frame. In MuJoCo, z comes
from ray casting; on the real robot, a flat terrain map is used as a placeholder.

Gamepad controls are the same as the Walk policy. If the robot shakes or destabilizes,
check `[AttentionDebug]` output: `grav` should be close to `[0, 0, -1]`, `scan_z` should
show the terrain height range, and action values should be within the training range.

## Sim2Real

### 1. Installation

```bash
conda activate legged_rl_lab
cd deploy/g1_deploy

python -m pip install cmake
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir -p build install
cd build
python -m cmake .. -DCMAKE_INSTALL_PREFIX=../install
python -m cmake --build . --target install -j"$(nproc)"

cd ~/legged_rl_lab
git submodule update --init --recursive deploy/g1_deploy/g1_python/unitree_sdk2_python

cd deploy/g1_deploy/g1_python/unitree_sdk2_python
export CYCLONEDDS_HOME=$(pwd)/../../cyclonedds/install
pip install -e .

pip install numpy==1.26.0 opencv-python==4.10.0.84 packaging==23.0 wheel==0.45.1
```

### 2. Robot Startup

Power on the G1 and keep it in zero-torque mode.

Enter debug mode with **L2 + R2**. The robot should be in damping mode. You can press
**L2 + A** to confirm debug mode, then press **L2 + R2** again to return to damping.

Safety shortcut: **L2 + B** enters damping mode immediately.

### 3. Network Setup

Connect the PC to the robot over Ethernet. Set the PC interface to the
`192.168.123.X` subnet. `192.168.123.99` is recommended.

Verify connectivity:

```bash
ping 192.168.123.161
```

### 4. Real-Robot Commands

Assume the Ethernet interface is `enp108s0`.

```bash
cd ~/legged_rl_lab/deploy/g1_deploy/g1_python
```

Walk:

```bash
python sim2real_walk.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_walk.yaml
```

AMP walk:

```bash
python sim2real_amp.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_amp.yaml \
  --model g1_walk.onnx
```

Mimic dance:

```bash
python sim2real_mimic.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_mimic.yaml \
  --model g1_dance.onnx
```

Attention:

```bash
python sim2real_attention.py \
  --net enp108s0 \
  --domain_id 0 \
  --config_path config/g1_attention.yaml \
  --model g1_attention2.onnx
```

For the real robot, the default `--terrain_source flat` fills the terrain map with a
flat floor at `init_height`. Use `--terrain_base_height` to override the flat-floor
reference height if the robot's standing height differs from the config default.

### 5. Real-Robot Operation

1. Start the controller. The robot begins in zero-torque mode.
2. Press **Start** on the remote. The robot moves to the default joint pose.
3. After the default pose is reached, lower the safety support until both feet touch the ground.
4. Press **A** to start motion control.
5. Lower support gradually after the output is stable.
6. Press **Select** to enter damping mode and exit.

Remote commands:

| Input | Function |
| --- | --- |
| Left stick forward/backward | X velocity |
| Left stick left/right | Y velocity |
| Right stick left/right | Yaw velocity |
| Select | Damping / exit |
