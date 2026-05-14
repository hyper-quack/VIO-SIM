import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
import math


class AGPFPlanner(Node):
    """
    Stabilized A-star Guided Potential Field local planner.

    Main fixes:
      1. d_A increased from 1.0 to 2.0 so the planner does not constantly
         fall into return-to-path mode.
      2. SAFE_DIST reduced from 1.0 to 0.4. The previous 1.0 m value made
         the drone overreact to walls and obstacles.
      3. Repulsive force is capped. Without a cap, many costmap cells can
         accumulate into a huge force and flip the velocity direction.
      4. Output velocity is low-pass filtered and acceleration-limited.
         This prevents yaw spinning in the mission node.
      5. Small lateral oscillations are damped when the path direction is clear.
    """

    # ------------------------------------------------------------------
    # AGPF parameters
    # ------------------------------------------------------------------
    K_A = 2.0
    d_A = 2.0              # was 1.0, too small after A* simplification
    d_G = 1.5

    K_O = 1.4              # was 2.0, too aggressive with dense costmap
    l_O = 1.5
    d_O = 1.5

    K_V = 0.20             # was 0.5, reduced because odom velocity can be noisy

    # ------------------------------------------------------------------
    # Safety / repulsion
    # ------------------------------------------------------------------
    SAFE_DIST = 0.4        # was 1.0, caused strong oscillation
    SAFE_DIST_GAIN = 4.0

    REP_FORCE_MAX = 1.2    # cap total repulsive force
    TOTAL_FORCE_MAX = 3.0  # cap total AGPF force before velocity conversion

    # Ignore obstacle cells unrealistically close to drone center.
    # This protects against residual self-detection in costmap.
    SELF_OBS_IGNORE_RADIUS = 0.35

    # ------------------------------------------------------------------
    # Output limits
    # ------------------------------------------------------------------
    MAX_VEL = 0.35         # slightly lower for stable yaw
    GOAL_REACHED_DIST = 0.3
    WAYPOINT_SKIP_DIST = 0.45

    # Velocity smoothing
    VEL_FILTER_ALPHA = 0.25       # lower = smoother output
    MAX_VEL_STEP =  0.05        # max velocity change per 20 Hz cycle
    YAW_DEADBAND_SPEED = 0.02     # below this, publish zero to avoid yaw jitter

    # Goal-reached cooldown
    GOAL_REACHED_COOLDOWN = 2.0

    def __init__(self):
        super().__init__('agpf_planner')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.global_path = []
        self.current_pos = None
        self.current_z = -2.0
        self.last_pos = None
        self.last_pos_time = None
        self.current_vel = np.zeros(2)

        # Filtered output velocity
        self.filtered_velocity = np.zeros(2)

        # Sensor distances
        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')

        # Costmap
        self.costmap = None
        self.costmap_resolution = 0.05
        self.costmap_origin = (0.0, 0.0)
        self.costmap_w = 100
        self.costmap_h = 100

        # Path progress
        self.current_wp_idx = 0

        # Goal tracking
        self.current_goal = None
        self.goal_reached = False
        self.goal_reached_time = 0.0

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/current_pose', self.pose_callback, qos)
        self.dist_sub = self.create_subscription(
            Float32MultiArray, '/obstacle_distances', self.distances_callback, qos)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/costmap', self.costmap_callback, 10)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self.vel_pub = self.create_publisher(
            TwistStamped, '/desired_velocity', 10)
        self.goal_reached_pub = self.create_publisher(
            Bool, '/goal_reached', 10)

        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            f'AGPF Planner STABILIZED started ✓  '
            f'd_A={self.d_A} SAFE_DIST={self.SAFE_DIST} '
            f'K_O={self.K_O} REP_MAX={self.REP_FORCE_MAX} '
            f'MAX_VEL={self.MAX_VEL}'
        )

    # ==================================================================
    # Callbacks
    # ==================================================================

    def path_callback(self, msg):
        new_path = [
            (p.pose.position.x, p.pose.position.y)
            for p in msg.poses
        ]

        if not new_path:
            return

        new_goal = new_path[-1]

        goal_changed = (
            self.current_goal is None or
            abs(new_goal[0] - self.current_goal[0]) > 0.1 or
            abs(new_goal[1] - self.current_goal[1]) > 0.1
        )

        self.global_path = new_path

        if goal_changed:
            self.current_wp_idx = 0
            self.goal_reached = False
            self.current_goal = new_goal
            self.filtered_velocity[:] = 0.0
            self.get_logger().info(
                f'New goal path -> ({new_goal[0]:.2f},{new_goal[1]:.2f})  '
                f'{len(new_path)} waypoints — idx reset'
            )
        else:
            # Same goal replan. Keep progress by selecting the closest waypoint
            # in the new path instead of restarting from index 0.
            self.current_wp_idx = self._closest_index_on_path(new_path)
            self.get_logger().info(
                f'Replan same goal — {len(new_path)} waypoints  '
                f'idx={self.current_wp_idx} goal_reached={self.goal_reached}'
            )

    def pose_callback(self, msg):
        new_pos = np.array([msg.pose.position.x, msg.pose.position.y], dtype=float)
        self.current_z = msg.pose.position.z
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_pos is not None and self.last_pos_time is not None:
            dt = now - self.last_pos_time
            if dt > 0.001:
                raw_vel = (new_pos - self.last_pos) / dt

                # Reject impossible velocity spikes from timestamp jitter.
                if np.linalg.norm(raw_vel) < 3.0:
                    self.current_vel = 0.85 * self.current_vel + 0.15 * raw_vel

        self.last_pos = new_pos
        self.last_pos_time = now
        self.current_pos = new_pos

    def distances_callback(self, msg):
        if len(msg.data) >= 3:
            self.front_dist = msg.data[0]
            self.left_dist = msg.data[1]
            self.right_dist = msg.data[2]

    def costmap_callback(self, msg):
        self.costmap = np.array(
            msg.data, dtype=np.int8
        ).reshape(msg.info.height, msg.info.width)

        self.costmap_resolution = msg.info.resolution
        self.costmap_origin = (
            msg.info.origin.position.x,
            msg.info.origin.position.y
        )
        self.costmap_w = msg.info.width
        self.costmap_h = msg.info.height

    # ==================================================================
    # Path helpers
    # ==================================================================

    def _closest_index_on_path(self, path):
        if self.current_pos is None or not path:
            return 0

        start = min(self.current_wp_idx, len(path) - 1)
        best_idx = start
        best_dist = float('inf')

        for i in range(start, len(path)):
            wp = np.array(path[i], dtype=float)
            d = np.linalg.norm(wp - self.current_pos)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _find_closest_waypoint(self):
        best_dist = float('inf')
        best_wp = None
        best_idx = self.current_wp_idx

        for i in range(self.current_wp_idx, len(self.global_path)):
            wp = np.array(self.global_path[i], dtype=float)
            dist = np.linalg.norm(wp - self.current_pos)
            if dist < best_dist:
                best_dist = dist
                best_wp = wp
                best_idx = i

        self.current_wp_idx = best_idx
        return best_wp, best_dist

    # ==================================================================
    # Force computation
    # ==================================================================

    def compute_attractive_force(self):
        if not self.global_path or self.current_pos is None:
            return np.zeros(2), 0

        f_att = np.zeros(2)
        used_count = 0

        # Skip waypoints already reached/passed
        while self.current_wp_idx < len(self.global_path) - 1:
            wp = np.array(self.global_path[self.current_wp_idx], dtype=float)
            if np.linalg.norm(wp - self.current_pos) < self.WAYPOINT_SKIP_DIST:
                self.current_wp_idx += 1
            else:
                break

        # Pull toward waypoints within d_A
        for i in range(self.current_wp_idx, len(self.global_path)):
            wp_pos = np.array(self.global_path[i], dtype=float)
            rho = wp_pos - self.current_pos
            dist = np.linalg.norm(rho)

            if 0.01 < dist <= self.d_A:
                # Weight nearer waypoints slightly more smoothly
                weight = 1.0 - 0.3 * (dist / self.d_A)
                f_att += self.K_A * weight * rho
                used_count += 1
            elif dist > self.d_A and used_count > 0:
                break

        # Fallback: closest unvisited waypoint
        if used_count == 0 and self.current_wp_idx < len(self.global_path):
            closest_wp, closest_dist = self._find_closest_waypoint()
            if closest_wp is not None and closest_dist > 0.01:
                rho = closest_wp - self.current_pos
                f_att = self.K_A * rho / max(closest_dist, 1.0)
                used_count = 1
                self.get_logger().warn(
                    f'Return-to-path — closest wp at {closest_dist:.2f} m',
                    throttle_duration_sec=1.0
                )

        return f_att, used_count

    def compute_repulsive_force(self, f_att):
        if self.costmap is None or self.current_pos is None:
            return np.zeros(2)

        f_att_norm = np.linalg.norm(f_att)
        if f_att_norm < 0.01:
            return self._sensor_based_repulsive()

        f_att_unit = f_att / f_att_norm
        f_rep_total = np.zeros(2)

        search_radius_cells = int(self.d_O / self.costmap_resolution)
        safe_cells = int(self.SAFE_DIST / self.costmap_resolution)

        cx = int((self.current_pos[0] - self.costmap_origin[0]) /
                 self.costmap_resolution)
        cy = int((self.current_pos[1] - self.costmap_origin[1]) /
                 self.costmap_resolution)

        if not (0 <= cx < self.costmap_w and 0 <= cy < self.costmap_h):
            return np.zeros(2)

        # Step 3 reduces dense-cell accumulation.
        # Dense costmaps otherwise create huge summed repulsion.
        for dx in range(-search_radius_cells, search_radius_cells + 1, 3):
            for dy in range(-search_radius_cells, search_radius_cells + 1, 3):
                gx = cx + dx
                gy = cy + dy

                if not (0 <= gx < self.costmap_w and 0 <= gy < self.costmap_h):
                    continue

                if self.costmap[gy, gx] < 50:
                    continue

                obs_x = gx * self.costmap_resolution + self.costmap_origin[0]
                obs_y = gy * self.costmap_resolution + self.costmap_origin[1]

                rho_o = self.current_pos - np.array([obs_x, obs_y], dtype=float)
                dist = np.linalg.norm(rho_o)

                if dist > self.d_O:
                    continue

                # Ignore very close cells. They are often residual self-map noise.
                if dist < self.SELF_OBS_IGNORE_RADIUS:
                    continue

                direction = rho_o / dist

                # Repulsive magnitude. Stronger near obstacle, softer far away.
                # This form is more stable than the old accumulated exponential force.
                magnitude = self.K_O * (1.0 / dist - 1.0 / self.d_O) / (dist * dist)
                magnitude = max(0.0, min(magnitude, 2.0))

                f_o = magnitude * direction

                cell_dist = math.sqrt(dx * dx + dy * dy)

                if cell_dist <= safe_cells:
                    # Inside safety band: full repulsion, but still capped later.
                    f_rep_total += self.SAFE_DIST_GAIN * f_o
                else:
                    # Outside safety band: orthogonal component only.
                    f_parallel = np.dot(f_o, f_att_unit) * f_att_unit
                    f_orthogonal = f_o - f_parallel
                    f_rep_total += f_orthogonal

        # Cap repulsive force to avoid velocity flipping/spinning.
        rep_norm = np.linalg.norm(f_rep_total)
        if rep_norm > self.REP_FORCE_MAX:
            f_rep_total = (f_rep_total / rep_norm) * self.REP_FORCE_MAX

        return f_rep_total

    def _sensor_based_repulsive(self):
        f = np.zeros(2)

        # This is only a fallback when no attractive force exists.
        # Keep it weak to avoid spinning.
        if math.isfinite(self.front_dist) and self.front_dist < self.d_O:
            mag = 0.4 * self.K_O / max(self.front_dist, 0.2)
            if self.front_dist < self.SAFE_DIST:
                mag *= self.SAFE_DIST_GAIN
            f[0] -= mag

        if math.isfinite(self.left_dist) and self.left_dist < self.d_O:
            mag = 0.4 * self.K_O / max(self.left_dist, 0.2)
            if self.left_dist < self.SAFE_DIST:
                mag *= self.SAFE_DIST_GAIN
            f[1] -= mag

        if math.isfinite(self.right_dist) and self.right_dist < self.d_O:
            mag = 0.4 * self.K_O / max(self.right_dist, 0.2)
            if self.right_dist < self.SAFE_DIST:
                mag *= self.SAFE_DIST_GAIN
            f[1] += mag

        f_norm = np.linalg.norm(f)
        if f_norm > self.REP_FORCE_MAX:
            f = (f / f_norm) * self.REP_FORCE_MAX

        return f

    def compute_velocity_force(self):
        v_norm = np.linalg.norm(self.current_vel)
        if v_norm < 0.05:
            return np.zeros(2)
        return self.K_V * self.current_vel / v_norm

    # ==================================================================
    # Output filtering
    # ==================================================================

    def _filter_velocity(self, desired_velocity):
        desired_norm = np.linalg.norm(desired_velocity)

        # Only stop if the desired command itself is tiny.
        # Do not kill the filtered velocity while it is ramping up.
        if desired_norm < self.YAW_DEADBAND_SPEED:
            self.filtered_velocity[:] = 0.0
            return self.filtered_velocity.copy()

        filtered = (
            (1.0 - self.VEL_FILTER_ALPHA) * self.filtered_velocity +
            self.VEL_FILTER_ALPHA * desired_velocity
        )

        delta = filtered - self.filtered_velocity
        delta_norm = np.linalg.norm(delta)

        if delta_norm > self.MAX_VEL_STEP:
            delta = delta / delta_norm * self.MAX_VEL_STEP
            filtered = self.filtered_velocity + delta

        speed = np.linalg.norm(filtered)
        if speed > self.MAX_VEL:
            filtered = filtered / speed * self.MAX_VEL

        self.filtered_velocity = filtered
        return filtered

    # ==================================================================
    # Main control loop
    # ==================================================================

    def control_loop(self):
        if self.current_pos is None or not self.global_path:
            self._publish_zero_velocity()
            return

        goal = np.array(self.global_path[-1], dtype=float)
        dist_to_goal = np.linalg.norm(goal - self.current_pos)

        if dist_to_goal < self.GOAL_REACHED_DIST:
            now = self.get_clock().now().nanoseconds * 1e-9
            if (not self.goal_reached and
                    (now - self.goal_reached_time) > self.GOAL_REACHED_COOLDOWN):
                self.get_logger().info(
                    f'Goal reached! ({goal[0]:.2f},{goal[1]:.2f})'
                )
                self.goal_reached = True
                self.goal_reached_time = now
                msg = Bool()
                msg.data = True
                self.goal_reached_pub.publish(msg)

            self._publish_zero_velocity()
            return

        f_att, n_used = self.compute_attractive_force()
        f_rep = self.compute_repulsive_force(f_att)
        f_v = self.compute_velocity_force()

        f_tot = f_att + f_rep + f_v

        # Cap total force
        f_norm = np.linalg.norm(f_tot)
        if f_norm > self.TOTAL_FORCE_MAX:
            f_tot = f_tot / f_norm * self.TOTAL_FORCE_MAX
            f_norm = self.TOTAL_FORCE_MAX

        if f_norm < 0.01:
            self._publish_zero_velocity()
            return

        desired_speed = min(f_norm, self.MAX_VEL)
        desired_velocity = (f_tot / f_norm) * desired_speed

        velocity = self._filter_velocity(desired_velocity)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = 0.0
        self.vel_pub.publish(msg)

        self.get_logger().info(
            f'AGPF stable — wp[{self.current_wp_idx}/{len(self.global_path)}] '
            f'pos:({self.current_pos[0]:.2f},{self.current_pos[1]:.2f}) '
            f'goal:{dist_to_goal:.2f}m '
            f'F_att:({f_att[0]:.2f},{f_att[1]:.2f}) '
            f'F_rep:({f_rep[0]:.2f},{f_rep[1]:.2f}) '
            f'F_v:({f_v[0]:.2f},{f_v[1]:.2f}) '
            f'vel:({velocity[0]:.2f},{velocity[1]:.2f})',
            throttle_duration_sec=1.0
        )

    def _publish_zero_velocity(self):
        self.filtered_velocity[:] = 0.0

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        self.vel_pub.publish(msg)


def main():
    rclpy.init()
    node = AGPFPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
