#!/usr/bin/env python3
"""
rl_depth_filter.py — RL-Based Depth Reliability Filter

Architecture:
  /oakd/depth/image          → depth frame (320×240)
  /imu/filtered              → yaw rate, linear acceleration
  /fmu/out/vehicle_odometry  → linear velocity
  /obstacle_distances        → front/left/right distances
  /planned_path              → path exists flag
        ↓
  Rule-based filter (Phase 1) OR PPO Agent (Phase 2)
        ↓
  /depth_trust_weight  → Float32 [0.0 - 1.0] → octomap_manager

Actions:
  0 → FULL_TRUST    (1.0) — clean frame, use fully
  1 → REDUCED_TRUST (0.6) — minor noise, partial use
  2 → LOW_TRUST     (0.2) — significant noise, minimal use
  3 → SKIP_FRAME    (0.0) — bad frame, ignore completely

Observation space (for PPO training):
  - depth image resized to 80×60 (normalized)
  - yaw_rate (rad/s)
  - linear velocity (m/s)
  - front/left/right distances (m)
  - new_obstacle_cells (count)
  - path_blocked (bool)
  - trust_history (last 5 actions)

Reward function (for PPO training):
  + progress_to_goal * 2.0
  + path_exists * 1.0
  + safe_distance_bonus * 0.5
  - new_false_obstacles * 0.01
  - emergency_stop * 0.5
  - collision * 5.0
  - excessive_skipping * 0.2

Training: PPO via Stable-Baselines3
  env = GazeboDepthFilterEnv(ros_node)
  model = PPO('CnnPolicy', env, verbose=1)
  model.learn(total_timesteps=500_000)
  model.save('rl_depth_filter_ppo')
"""

import math
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, String
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from px4_msgs.msg import VehicleOdometry
from cv_bridge import CvBridge

# ── Mode ──────────────────────────────────────────────────────────────────────
USE_RL_MODEL = False   # False = rule-based stub, True = PPO agent

# ── RL model path ─────────────────────────────────────────────────────────────
RL_MODEL_PATH = '/home/poorcsky/csky_ws/models/rl_depth_filter_ppo'

# ── Actions ───────────────────────────────────────────────────────────────────
ACTION_FULL_TRUST    = 0   # trust = 1.0
ACTION_REDUCED_TRUST = 1   # trust = 0.6
ACTION_LOW_TRUST     = 2   # trust = 0.2
ACTION_SKIP_FRAME    = 3   # trust = 0.0

TRUST_VALUES = {
    ACTION_FULL_TRUST:    1.0,
    ACTION_REDUCED_TRUST: 0.6,
    ACTION_LOW_TRUST:     0.2,
    ACTION_SKIP_FRAME:    0.0,
}

# ── Rule-based thresholds ─────────────────────────────────────────────────────
RULE_YAW_HIGH        = 0.40   # rad/s — skip frame
RULE_YAW_MEDIUM      = 0.20   # rad/s — low trust
RULE_YAW_LOW         = 0.10   # rad/s — reduced trust
RULE_VEL_HIGH        = 0.40   # m/s — reduced trust when moving fast
RULE_DIST_CLOSE      = 0.80   # m — low trust when very close to obstacle
RULE_SKIP_RATIO_MAX  = 0.60   # never skip more than 60% of frames

# ── Observation parameters ────────────────────────────────────────────────────
OBS_IMG_W  = 80    # resized depth image width
OBS_IMG_H  = 60    # resized depth image height
OBS_HIST   = 5     # number of past actions to include in observation

# ── Statistics window ─────────────────────────────────────────────────────────
STATS_WINDOW = 100  # frames for rolling statistics


class RLDepthFilter(Node):

    def __init__(self):
        super().__init__('rl_depth_filter')

        # ── QoS ───────────────────────────────────────────────────────────────
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        qos_imu = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Image, '/oakd/depth/image',
            self._depth_cb, qos_image)

        self.create_subscription(
            Imu, '/imu/filtered',
            self._imu_cb, qos_imu)

        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, qos_px4)

        self.create_subscription(
            Float32MultiArray, '/obstacle_distances',
            self._distances_cb, qos_reliable)

        self.create_subscription(
            Path, '/planned_path',
            self._path_cb, qos_reliable)

        # ── Publishers ────────────────────────────────────────────────────────
        self.trust_pub = self.create_publisher(
            Float32, '/depth_trust_weight', qos_reliable)

        self.debug_pub = self.create_publisher(
            String, '/rl_filter_debug', qos_reliable)

        # ── CV bridge ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── State ─────────────────────────────────────────────────────────────
        self.yaw_rate      = 0.0
        self.linear_vel    = 0.0
        self.front_dist    = float('inf')
        self.left_dist     = float('inf')
        self.right_dist    = float('inf')
        self.path_exists   = False
        self.last_depth    = None

        # Action history for observation
        self.action_history = [ACTION_FULL_TRUST] * OBS_HIST

        # Rolling statistics
        self.action_counts  = {a: 0 for a in range(4)}
        self.frame_count    = 0
        self.skip_count     = 0
        self.stats_window   = []

        # Current trust weight
        self.current_trust  = 1.0

        # ── RL model ──────────────────────────────────────────────────────────
        self.rl_model = None
        if USE_RL_MODEL:
            self._load_rl_model()

        mode = 'PPO agent' if (USE_RL_MODEL and self.rl_model) else 'rule-based'
        self.get_logger().info(f'RL Depth Filter started ✓ — mode: {mode}')
        self.get_logger().info(
            f'Publishing /depth_trust_weight at each depth frame')

    # ═════════════════════════════════════════════════════════════════════════
    # RL model loading
    # ═════════════════════════════════════════════════════════════════════════

    def _load_rl_model(self):
        """Load trained PPO model if available."""
        try:
            from stable_baselines3 import PPO
            import os
            if os.path.exists(RL_MODEL_PATH + '.zip'):
                self.rl_model = PPO.load(RL_MODEL_PATH)
                self.get_logger().info(f'PPO model loaded: {RL_MODEL_PATH}')
            else:
                self.get_logger().warn(
                    f'PPO model not found at {RL_MODEL_PATH} — using rules')
                self.rl_model = None
        except ImportError:
            self.get_logger().warn(
                'stable_baselines3 not installed — using rule-based filter')
            self.rl_model = None

    # ═════════════════════════════════════════════════════════════════════════
    # Subscribers
    # ═════════════════════════════════════════════════════════════════════════

    def _imu_cb(self, msg: Imu):
        self.yaw_rate = abs(float(msg.angular_velocity.z))

    def _odom_cb(self, msg: VehicleOdometry):
        vx = float(msg.velocity[0])
        vy = float(msg.velocity[1])
        self.linear_vel = math.sqrt(vx*vx + vy*vy)

    def _distances_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.front_dist = float(msg.data[0])
            self.left_dist  = float(msg.data[1])
            self.right_dist = float(msg.data[2])

    def _path_cb(self, msg: Path):
        self.path_exists = len(msg.poses) > 0

    # ═════════════════════════════════════════════════════════════════════════
    # Main depth callback
    # ═════════════════════════════════════════════════════════════════════════

    def _depth_cb(self, msg: Image):
        """
        Process each depth frame:
        1. Build observation vector
        2. Choose action (rule-based or PPO)
        3. Publish trust weight
        """
        self.frame_count += 1

        # Convert depth image
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f'depth_cb: {e}', throttle_duration_sec=2.0)
            return

        self.last_depth = depth

        # Build observation
        obs = self._build_observation(depth)

        # Choose action
        if USE_RL_MODEL and self.rl_model is not None:
            action = self._rl_action(obs)
        else:
            action = self._rule_based_action()

        # Anti-skip guard: never skip more than RULE_SKIP_RATIO_MAX frames
        if action == ACTION_SKIP_FRAME:
            recent = self.stats_window[-20:] if len(self.stats_window) >= 20 else self.stats_window
            if recent:
                skip_ratio = sum(1 for a in recent if a == ACTION_SKIP_FRAME) / len(recent)
                if skip_ratio >= RULE_SKIP_RATIO_MAX:
                    action = ACTION_LOW_TRUST
                    self.get_logger().info(
                        'Anti-skip guard: forced LOW_TRUST',
                        throttle_duration_sec=2.0)

        # Update history
        self.action_history.pop(0)
        self.action_history.append(action)
        self.action_counts[action] += 1
        self.stats_window.append(action)
        if len(self.stats_window) > STATS_WINDOW:
            self.stats_window.pop(0)

        # Get trust value
        trust = TRUST_VALUES[action]
        self.current_trust = trust

        # Publish
        trust_msg = Float32()
        trust_msg.data = float(trust)
        self.trust_pub.publish(trust_msg)

        # Debug
        self._publish_debug(action, trust)

    # ═════════════════════════════════════════════════════════════════════════
    # Observation builder
    # ═════════════════════════════════════════════════════════════════════════

    def _build_observation(self, depth: np.ndarray) -> dict:
        """
        Build observation dict for RL agent.
        Also used for logging/debugging in rule-based mode.
        """
        # Resize depth image to OBS_IMG_H × OBS_IMG_W
        depth_small = cv2.resize(depth, (OBS_IMG_W, OBS_IMG_H))

        # Normalize depth: 0=close(0.3m), 1=far(8m), nan→1.0
        depth_norm = np.clip(depth_small / 8.0, 0.0, 1.0)
        depth_norm = np.nan_to_num(depth_norm, nan=1.0)

        # Scalar observations
        scalars = np.array([
            min(self.yaw_rate / 1.0, 1.0),              # yaw rate normalized
            min(self.linear_vel / 1.0, 1.0),            # velocity normalized
            min(self.front_dist / 5.0, 1.0),            # front distance
            min(self.left_dist  / 5.0, 1.0),            # left distance
            min(self.right_dist / 5.0, 1.0),            # right distance
            1.0 if self.path_exists else 0.0,            # path exists
        ], dtype=np.float32)

        # Action history (one-hot not needed — just normalized action index)
        history = np.array(
            [a / 3.0 for a in self.action_history],
            dtype=np.float32)

        return {
            'depth_image': depth_norm.astype(np.float32),
            'scalars':     scalars,
            'history':     history,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Rule-based filter (Phase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _rule_based_action(self) -> int:
        """
        Hand-crafted rules for depth frame reliability.

        Priority order (highest to lowest):
        1. High yaw rate → skip frame (motion blur certain)
        2. Medium yaw rate → low trust
        3. Very close obstacle → low trust (near-field noise)
        4. Fast forward motion → reduced trust
        5. Low yaw, slow motion → full trust
        """

        # Rule 1 — High yaw rate: definitely bad frame
        if self.yaw_rate > RULE_YAW_HIGH:
            return ACTION_SKIP_FRAME

        # Rule 2 — Medium yaw rate: significant noise
        if self.yaw_rate > RULE_YAW_MEDIUM:
            return ACTION_LOW_TRUST

        # Rule 3 — Very close to obstacle: near-field depth unreliable
        min_dist = min(self.front_dist, self.left_dist, self.right_dist)
        if min_dist < RULE_DIST_CLOSE:
            return ACTION_LOW_TRUST

        # Rule 4 — Low yaw but fast motion: some blur possible
        if self.yaw_rate > RULE_YAW_LOW or self.linear_vel > RULE_VEL_HIGH:
            return ACTION_REDUCED_TRUST

        # Rule 5 — Stable: full trust
        return ACTION_FULL_TRUST

    # ═════════════════════════════════════════════════════════════════════════
    # PPO agent action (Phase 2)
    # ═════════════════════════════════════════════════════════════════════════

    def _rl_action(self, obs: dict) -> int:
        """
        Use trained PPO model to select action.
        Falls back to rule-based if model fails.
        """
        try:
            # Flatten observation for SB3
            flat_obs = np.concatenate([
                obs['depth_image'].flatten(),
                obs['scalars'],
                obs['history'],
            ])
            action, _ = self.rl_model.predict(flat_obs, deterministic=True)
            return int(action)
        except Exception as e:
            self.get_logger().warn(
                f'RL predict failed: {e} — rule fallback',
                throttle_duration_sec=2.0)
            return self._rule_based_action()

    # ═════════════════════════════════════════════════════════════════════════
    # Debug publisher
    # ═════════════════════════════════════════════════════════════════════════

    def _publish_debug(self, action: int, trust: float):
        action_names = {
            ACTION_FULL_TRUST:    'FULL',
            ACTION_REDUCED_TRUST: 'REDUCED',
            ACTION_LOW_TRUST:     'LOW',
            ACTION_SKIP_FRAME:    'SKIP',
        }

        # Rolling skip ratio
        recent = self.stats_window[-20:] if len(self.stats_window) >= 20 else self.stats_window
        skip_ratio = (sum(1 for a in recent if a == ACTION_SKIP_FRAME) /
                      max(len(recent), 1))

        debug = String()
        debug.data = (
            f'action={action_names[action]} '
            f'trust={trust:.1f} '
            f'yaw={self.yaw_rate:.2f}r/s '
            f'vel={self.linear_vel:.2f}m/s '
            f'front={self.front_dist:.2f}m '
            f'skip_ratio={skip_ratio:.2f} '
            f'frames={self.frame_count} '
            f'[F={self.action_counts[0]} '
            f'R={self.action_counts[1]} '
            f'L={self.action_counts[2]} '
            f'S={self.action_counts[3]}]'
        )
        self.debug_pub.publish(debug)

        self.get_logger().info(debug.data, throttle_duration_sec=2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Training Environment (for future use)
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_ENV_STUB = '''
"""
To train the PPO agent, use this environment stub with Stable-Baselines3.

Requirements:
    pip install stable-baselines3 gymnasium --break-system-packages

Training script (run separately, not as ROS node):

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class DepthFilterEnv(gym.Env):
    """
    Gymnasium environment for RL depth filter training.
    Connects to running ROS 2 stack via topic subscriptions.
    """

    def __init__(self):
        super().__init__()

        # Action space: 4 discrete actions
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: depth image + scalars + history
        img_size = 80 * 60
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(img_size + 6 + 5,),
            dtype=np.float32)

        self.current_obs = np.zeros(img_size + 6 + 5, dtype=np.float32)
        self.prev_goal_dist = float('inf')
        self.step_count = 0

    def reset(self, seed=None):
        # Reset Gazebo simulation
        # Call: ros2 service call /reset_simulation std_srvs/srv/Empty
        self.step_count = 0
        self.prev_goal_dist = float('inf')
        return self.current_obs, {}

    def step(self, action):
        # 1. Publish trust weight based on action
        trust = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}[action]
        # publish to /depth_trust_weight

        # 2. Wait one frame
        import time; time.sleep(0.1)

        # 3. Compute reward
        reward = self._compute_reward(action)

        # 4. Check done
        done = self.step_count > 1000 or self._collision_detected()

        self.step_count += 1
        return self.current_obs, reward, done, False, {}

    def _compute_reward(self, action):
        reward = 0.0

        # Progress toward goal
        goal_dist = self._get_goal_distance()
        progress = self.prev_goal_dist - goal_dist
        reward += 2.0 * progress
        self.prev_goal_dist = goal_dist

        # Path exists
        reward += 1.0 if self._path_exists() else -1.0

        # Safe distance
        min_dist = self._get_min_obstacle_dist()
        reward += 0.5 if min_dist > 1.0 else -0.5

        # Penalize excessive skipping
        if action == 3:  # SKIP
            reward -= 0.2

        # Collision
        if self._collision_detected():
            reward -= 5.0

        return reward

# Train:
env = DepthFilterEnv()
model = PPO(
    'MlpPolicy', env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log='./ppo_depth_filter_tb/')
model.learn(total_timesteps=500_000)
model.save('/home/poorcsky/csky_ws/models/rl_depth_filter_ppo')
print('Training complete — model saved')
"""
'''


def main(args=None):
    rclpy.init(args=args)
    node = RLDepthFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()