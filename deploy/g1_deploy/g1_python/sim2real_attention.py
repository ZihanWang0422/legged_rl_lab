#!/usr/bin/env python3
"""SDK2 sim2real controller for the G1 attention terrain policy.

Uses a flat terrain map (no real height scanner on the robot) to provide the
same observation structure the attention policy expects during training.

Usage:
    python deploy/g1_deploy/g1_python/sim2real_attention.py --net enp108s0 --config config/g1_attention.yaml --model g1_attention2.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
import yaml

THIS_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = THIS_DIR.parent
SDK2_PYTHON_DIR = THIS_DIR / "unitree_sdk2_python"
if str(SDK2_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(SDK2_PYTHON_DIR))


def _bootstrap_cyclonedds_home() -> None:
    """Ensure CYCLONEDDS_HOME points to a valid local install before SDK import."""
    env_home = os.environ.get("CYCLONEDDS_HOME", "")
    if env_home and (Path(env_home) / "lib" / "libddsc.so").exists():
        return

    for candidate in (THIS_DIR / "cyclonedds" / "install", DEPLOY_DIR / "cyclonedds" / "install"):
        if (candidate / "lib" / "libddsc.so").exists():
            os.environ["CYCLONEDDS_HOME"] = str(candidate)
            print(f"[CycloneDDS] Using CYCLONEDDS_HOME={candidate}")
            return


_bootstrap_cyclonedds_home()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from common.command_helper import MotorMode, create_damping_cmd, init_cmd_go, init_cmd_hg
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from config import Config


def resolve_config(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "config":
        return DEPLOY_DIR / candidate
    return DEPLOY_DIR / "config" / path


def resolve_model(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "exported_policy":
        return DEPLOY_DIR / candidate
    return DEPLOY_DIR / "exported_policy" / name


# ---------------------------------------------------------------------------
# Observation builders  (shared with sim2sim_attention.py)
# ---------------------------------------------------------------------------

def build_proprio_obs(
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    commands: np.ndarray,
    dof_pos_rel: np.ndarray,
    dof_vel: np.ndarray,
    last_action: np.ndarray,
    config: Config,
) -> np.ndarray:
    """
    Build one proprioceptive observation frame matching the C++ build_proprio_obs.

    Layout (policy/Isaac order):
      base_ang_vel(3)     * ang_vel_scale
      projected_gravity(3)
      commands(3)          * command_scale
      dof_pos_rel(N)       * dof_pos_scale
      dof_vel(N)           * dof_vel_scale
      last_action(N)       * last_action_scale
    Total: 9 + 3*N dims  (96 for 29-DoF G1).
    """
    return np.concatenate(
        [
            base_ang_vel * float(config.ang_vel_scale),
            projected_gravity,
            commands * np.asarray(config.command_scale, dtype=np.float32),
            dof_pos_rel * float(config.dof_pos_scale),
            dof_vel * float(config.dof_vel_scale),
            last_action * float(getattr(config, "last_action_scale", 1.0)),
        ]
    ).astype(np.float32)


def build_flat_terrain_map(config: Config, body_height: float) -> np.ndarray:
    """
    Build a flat terrain map that mimics the scan grid layout.

    The real robot has no terrain scanner; we fill every grid point with
    z = -body_height (the expected height of a flat floor below the pelvis),
    then clip to [z_min, z_max].

    Returns a flat float32 array of length terrain_map_length * terrain_map_width * coord_dim.
    """
    length = int(config.terrain_map_length)
    width = int(config.terrain_map_width)
    coord_dim = int(config.terrain_map_coord_dim)
    if coord_dim != 3:
        raise ValueError(f"terrain_map_coord_dim={coord_dim}, expected 3.")

    size_x, size_y = np.asarray(config.terrain_map_size, dtype=np.float32)
    xs = np.linspace(-size_x / 2.0, size_x / 2.0, length, dtype=np.float32)
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, width, dtype=np.float32)

    ordering = getattr(config, "terrain_ordering", "xy")
    if ordering == "yx":
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
    else:
        xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # All z = -body_height (flat floor at body height below pelvis)
    zz = np.full_like(xx, -float(body_height), dtype=np.float32)
    z_min, z_max = np.asarray(config.terrain_map_z_clip, dtype=np.float32)
    zz = np.clip(zz, z_min, z_max)

    return np.stack([xx, yy, zz], axis=-1).reshape(-1).astype(np.float32)


def load_attention_config(path: Path) -> Config:
    """
    Load the base Config from YAML and attach attention-specific fields that
    are not covered by the generic Config constructor.
    """
    config = Config(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    for key in (
        "terrain_map_length",
        "terrain_map_width",
        "terrain_map_coord_dim",
        "terrain_map_resolution",
        "terrain_map_size",
        "terrain_ordering",
        "terrain_map_z_clip",
        "init_height",
        "last_action_scale",
    ):
        if key in raw:
            setattr(config, key, raw[key])
    return config


# ---------------------------------------------------------------------------
# Attention controller (sim2real over SDK2 / CycloneDDS)
# ---------------------------------------------------------------------------

class AttentionController:
    """SDK2 sim2real controller for the G1 attention terrain policy."""

    def __init__(
        self,
        config: Config,
        net: str,
        domain_id: int = 0,
        debug_policy: bool = False,
        terrain_source: str = "flat",
        terrain_base_height: float | None = None,
    ) -> None:
        if terrain_source != "flat":
            raise ValueError(
                "Only --terrain_source flat is currently supported for sim2real attention."
            )

        ChannelFactoryInitialize(domain_id, net)

        self.config = config
        self.debug_policy = debug_policy
        self._last_debug_time = 0.0
        self.control_start_time = 0.0
        self.remote_controller = RemoteController()

        # ONNX policy
        self.policy = ort.InferenceSession(
            str(config.policy_path), providers=["CPUExecutionProvider"]
        )
        self.policy_input_names = [inp.name for inp in self.policy.get_inputs()]
        self.policy_output_names = [out.name for out in self.policy.get_outputs()]

        # Recurrent threads
        self.run_thread = RecurrentThread(
            interval=self.config.control_dt, target=self.run
        )
        self.publish_thread = RecurrentThread(interval=1 / 500, target=self.publish)
        self.cmd_lock = Lock()

        # State buffers
        self.joint_pos = np.zeros(config.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)

        # Terrain map (flat, pre-computed once — real robot has no scanner)
        body_height = float(
            terrain_base_height
            if terrain_base_height is not None
            else getattr(config, "init_height", 0.9)
        )
        self.terrain_map = build_flat_terrain_map(config, body_height)

        # Observation dimensions
        # proprio = base_ang_vel(3) + proj_gravity(3) + commands(3) + joint_pos(N) + joint_vel(N) + last_action(N)
        proprio_dim = 9 + 3 * config.num_actions
        terrain_dim = self.terrain_map.size
        expected_obs_dim = proprio_dim + terrain_dim

        self._proprio_dim = proprio_dim
        self._terrain_dim = terrain_dim
        self._expected_obs_dim = expected_obs_dim

        self._validate_dimensions()

        # Command limits
        self.command_min = np.array(
            [
                config.command_range["lin_vel_x"][0],
                config.command_range["lin_vel_y"][0],
                config.command_range["ang_vel_z"][0],
            ],
            dtype=np.float32,
        )
        self.command_max = np.array(
            [
                config.command_range["lin_vel_x"][1],
                config.command_range["lin_vel_y"][1],
                config.command_range["ang_vel_z"][1],
            ],
            dtype=np.float32,
        )
        self.command_slew_rate = np.asarray(
            getattr(config, "gamepad_slew_rate", config.command_rate_limit),
            dtype=np.float32,
        )

        # DDS channels
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowstate_subscriber = ChannelSubscriber(
                config.lowstate_topic, LowStateHG
            )
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowstate_subscriber = ChannelSubscriber(
                config.lowstate_topic, LowStateGo
            )
        else:
            raise ValueError("Invalid msg_type")

        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.wait_for_low_state()
        if config.msg_type == "hg":
            self.low_cmd = init_cmd_hg(
                self.low_cmd, self.mode_machine_, self.mode_pr_
            )
        else:
            self.low_cmd = init_cmd_go(
                self.low_cmd, weak_motor=self.config.weak_motor
            )

        self.publish_thread.Start()
        self.wait_for_start()
        self.move_to_default_pos()
        self.wait_for_control()

        print("Start Attention Control!")
        self.control_start_time = time.time()
        self.run_thread.Start()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dimensions(self) -> None:
        """Cross-check YAML num_obs / num_actions against ONNX shapes."""
        input_spec = self.policy.get_inputs()[0]
        output_spec = self.policy.get_outputs()[0]
        input_dim = (
            int(input_spec.shape[-1])
            if isinstance(input_spec.shape[-1], (int, np.integer))
            else None
        )
        output_dim = (
            int(output_spec.shape[-1])
            if isinstance(output_spec.shape[-1], (int, np.integer))
            else None
        )

        if int(self.config.num_obs) != self._expected_obs_dim:
            raise ValueError(
                f"num_obs={self.config.num_obs}, expected {self._expected_obs_dim} "
                f"(proprio={self._proprio_dim} + terrain={self._terrain_dim})."
            )
        if input_dim is not None and input_dim != self._expected_obs_dim:
            raise ValueError(
                f"ONNX input dim={input_dim}, expected {self._expected_obs_dim}."
            )
        if output_dim is not None and output_dim != self.config.num_actions:
            raise ValueError(
                f"ONNX output dim={output_dim}, expected {self.config.num_actions}."
            )

        print(
            f"[Check] Attention obs: proprio={self._proprio_dim}, "
            f"terrain={self._terrain_dim}, total={self._expected_obs_dim}, "
            f"onnx_input={input_dim}, actions={output_dim}"
        )

    # ------------------------------------------------------------------
    # DDS callbacks & helpers
    # ------------------------------------------------------------------

    def LowStateHandler(self, msg):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def publish(self):
        with self.cmd_lock:
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)

    def stop(self):
        print("Select Button detected, Exit!")
        self.publish_thread.Wait()
        with self.cmd_lock:
            self.low_cmd = create_damping_cmd(self.low_cmd)
            self.low_cmd.crc = CRC().Crc(self.low_cmd)
            self.lowcmd_publisher_.Write(self.low_cmd)
        time.sleep(0.2)
        sys.exit(0)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        self.mode_machine_ = self.low_state.mode_machine
        print("Successfully connected to the robot.")

    def wait_for_start(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal to move to default pos...")
        while self.remote_controller.button[KeyMap.start] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)
        init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
            init_dof_pos[i] = self.low_state.motor_state[motor_idx].q

        for i in range(num_step):
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            alpha = i / num_step
            with self.cmd_lock:
                for j, motor_idx in enumerate(self.config.sdk2isaac_idx):
                    self.low_cmd.motor_cmd[motor_idx].q = float(
                        init_dof_pos[j] * (1.0 - alpha)
                        + self.config.default_joint_pos[j] * alpha
                    )
                    self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                    self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[j])
                    self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[j])
                    self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            time.sleep(self.config.control_dt)

    def wait_for_control(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal to Start Attention Control...")
        while self.remote_controller.button[KeyMap.A] != 1:
            if self.remote_controller.button[KeyMap.select] == 1:
                self.stop()
            time.sleep(self.config.control_dt)

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def _remote_axes_to_command(self) -> np.ndarray:
        """Convert gamepad axes to velocity command in physical units (m/s, rad/s).

        Command scaling (command_scale) is applied later in build_proprio_obs,
        matching the C++ code path.
        """
        axes = np.array(
            [
                self.remote_controller.ly,
                -self.remote_controller.lx,
                -self.remote_controller.rx,
            ],
            dtype=np.float32,
        )
        if self.config.command_deadband > 0.0:
            axes[np.abs(axes) < self.config.command_deadband] = 0.0

        command = np.where(
            axes >= 0.0,
            axes * self.command_max,
            -(-axes) * np.abs(self.command_min),
        )
        return np.clip(command, self.command_min, self.command_max).astype(np.float32)

    def update_command(self, command_raw: np.ndarray) -> np.ndarray:
        """Apply slew-rate limiting and optional smoothing.

        The returned command is still in physical units (m/s, rad/s).
        command_scale is applied inside build_proprio_obs.
        """
        max_delta = self.command_slew_rate * self.config.control_dt
        delta = np.clip(command_raw - self.command, -max_delta, max_delta)
        self.command = self.command + delta
        if self.config.command_smoothing_tau > 0.0:
            alpha = self.config.control_dt / (
                self.config.command_smoothing_tau + self.config.control_dt
            )
            self.command = self.command + alpha * (command_raw - self.command)
        return self.command.copy()

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def build_attention_obs(self) -> np.ndarray:
        """Build the full attention observation: proprioceptive + terrain map.

        Returns flat (proprio_dim + terrain_dim) float32 array.
        """
        return np.concatenate(
            [self._current_proprio, self.terrain_map]
        ).astype(np.float32)

    def infer_policy(self, obs_batch: np.ndarray) -> np.ndarray:
        outputs = self.policy.run(
            self.policy_output_names,
            {self.policy_input_names[0]: obs_batch.astype(np.float32)},
        )
        return outputs[0].flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Main control loop (called at control_dt by RecurrentThread)
    # ------------------------------------------------------------------

    def run(self):
        # --- Read joint state ---
        for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
            self.joint_pos[i] = self.low_state.motor_state[motor_idx].q
            self.joint_vel[i] = self.low_state.motor_state[motor_idx].dq

        # --- IMU / base state ---
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.torso_idx].q
            waist_yaw_omega = self.low_state.motor_state[self.config.torso_idx].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel.reshape(1, 3),
            )
            ang_vel = np.asarray(ang_vel, dtype=np.float32).reshape(3)

        projected_gravity = get_gravity_orientation(quat)
        joint_pos_rel = self.joint_pos - self.config.default_joint_pos
        command = self.update_command(self._remote_axes_to_command())

        # --- Build proprioceptive observation ---
        self._current_proprio = build_proprio_obs(
            ang_vel,
            projected_gravity,
            command,
            joint_pos_rel,
            self.joint_vel,
            self.action,
            self.config,
        )

        # --- Full observation (proprio + flat terrain map) ---
        obs_batch = self.build_attention_obs().reshape(1, -1).astype(np.float32)

        # --- Policy inference ---
        raw_action = self.infer_policy(obs_batch)
        action_clip = getattr(self.config, "action_clip", None)
        if action_clip is not None:
            raw_action = np.clip(
                raw_action, -float(action_clip), float(action_clip)
            ).astype(np.float32)
        self.action = raw_action.astype(np.float32)

        # --- Compute target joint positions ---
        target_dof_pos = (
            self.config.default_joint_pos + self.action * self.config.action_scale
        )

        # Policy ramp (soft start)
        ramp = 1.0
        if self.config.policy_ramp_time > 0.0:
            ramp = np.clip(
                (time.time() - self.control_start_time)
                / self.config.policy_ramp_time,
                0.0,
                1.0,
            )
            target_dof_pos = self.config.default_joint_pos + ramp * (
                target_dof_pos - self.config.default_joint_pos
            )

        self._print_policy_debug(
            command, projected_gravity, ang_vel, joint_pos_rel, target_dof_pos, ramp
        )

        # --- Send commands ---
        with self.cmd_lock:
            for i, motor_idx in enumerate(self.config.sdk2isaac_idx):
                self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])

    # ------------------------------------------------------------------
    # Debug printing
    # ------------------------------------------------------------------

    def _print_policy_debug(
        self,
        command: np.ndarray,
        projected_gravity: np.ndarray,
        ang_vel: np.ndarray,
        joint_pos_rel: np.ndarray,
        target_dof_pos: np.ndarray,
        ramp: float,
    ) -> None:
        if not self.debug_policy or time.time() - self._last_debug_time < 0.5:
            return

        terrain_xyz = self.terrain_map.reshape(-1, 3)
        print(
            "[AttentionDebug] "
            f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
            f"grav=[{projected_gravity[0]:+.2f},{projected_gravity[1]:+.2f},{projected_gravity[2]:+.2f}] "
            f"ang=[{ang_vel[0]:+.2f},{ang_vel[1]:+.2f},{ang_vel[2]:+.2f}] "
            f"q_rel=[{joint_pos_rel.min():+.2f},{joint_pos_rel.max():+.2f}] "
            f"scan_z=[{terrain_xyz[:, 2].min():+.2f},{terrain_xyz[:, 2].max():+.2f}] "
            f"ramp={ramp:.2f} action=[{self.action.min():+.2f},{self.action.max():+.2f}] "
            f"target=[{target_dof_pos.min():+.2f},{target_dof_pos.max():+.2f}]"
        )
        self._last_debug_time = time.time()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDK2 sim2real controller for the G1 attention terrain policy."
    )
    parser.add_argument(
        "--net", type=str, default="enp108s0", help="network interface"
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=0,
        help="DDS domain id, use 1 for local SDK2 MuJoCo bridge",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/g1_attention.yaml",
        help="configuration file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="g1_attention2.onnx",
        help="ONNX model filename or path",
    )
    parser.add_argument(
        "--debug_policy",
        action="store_true",
        help="Print policy observation/action ranges.",
    )
    parser.add_argument(
        "--terrain_source",
        choices=["flat"],
        default="flat",
        help="Terrain map source for attention obs.",
    )
    parser.add_argument(
        "--terrain_base_height",
        type=float,
        default=None,
        help="Flat terrain z is filled as -height before clipping.",
    )
    args = parser.parse_args()

    config = load_attention_config(resolve_config(args.config_path))
    config.set_policy_path(str(resolve_model(args.model)))
    controller = AttentionController(
        config,
        args.net,
        args.domain_id,
        debug_policy=args.debug_policy,
        terrain_source=args.terrain_source,
        terrain_base_height=args.terrain_base_height,
    )

    try:
        while True:
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("Select Button detected, Exit!")
                break
            time.sleep(0.01)
    finally:
        controller.run_thread.Wait()
        controller.publish_thread.Wait()
        with controller.cmd_lock:
            controller.low_cmd = create_damping_cmd(controller.low_cmd)
            controller.low_cmd.crc = CRC().Crc(controller.low_cmd)
            controller.lowcmd_publisher_.Write(controller.low_cmd)
        time.sleep(0.2)
        print("Exit")


if __name__ == "__main__":
    main()
