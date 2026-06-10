# CSKy Drone — Navigation System Architecture

Generated from a complete read of every source file in:
`mission_manager`, `navigation_manager`, `flight_controller_bridge`, `vio_localization`.

> Purpose of this document: reference for debugging **why the drone loses position during flight.**
> The most likely causes are collected in §5 (Known Issues), with the prime suspects flagged ⛔.

---

## 0. COORDINATE CONVENTIONS (the single most important section)

Every node hand-rolls the same transforms. The convention is:

| Frame | Axes |
|-------|------|
| **PX4 local (NED)** | `position[0]=North`, `position[1]=East`, `position[2]=Down` |
| **World ("odom")** | `X = PX4 East`, `Y = PX4 North`, `Z = Up` |
| **Spawn offset** | World = PX4 + `(SPAWN_X=1.0, SPAWN_Y=3.0)` |

Canonical World mapping (repeated in ~7 files):
```
world_x = position[1] + 1.0     # East  + SPAWN_X
world_y = position[0] + 3.0     # North + SPAWN_Y
world_z = -position[2]          # Up
world_yaw = atan2(siny,cosy) - pi/2     # NED yaw → ENU/world yaw
```
Files doing this: [mission_node.py:210-212](src/mission_manager/mission_manager/mission_node.py#L210-L212), [octomap_manager.py:220-232](src/navigation_manager/navigation_manager/octomap_manager.py#L220-L232), [obstacle_detector.py:177-185](src/navigation_manager/navigation_manager/obstacle_detector.py#L177-L185), [pose_fusion.py:104-110](src/navigation_manager/navigation_manager/pose_fusion.py#L104-L110), [tf_odom_base.py:77-86](src/navigation_manager/navigation_manager/tf_odom_base.py#L77-L86), [odom_to_path.py:96-97](src/vio_localization/vio_localization/odom_to_path.py#L96-L97).

PX4 yaw convention: `yaw=0 → North`, `yaw=pi/2 → East`, so `target_yaw = atan2(East_vel, North_vel) = atan2(world_vx, world_vy)` ([mission_node.py:389](src/mission_manager/mission_manager/mission_node.py#L389)).

---

## 1. COMPLETE TOPIC GRAPH

### PX4 / FMU interface topics

| Topic | Type | Publisher (file:line) | Subscribers (file:line) | Data flow |
|-------|------|------------------------|--------------------------|-----------|
| `/fmu/in/offboard_control_mode` | `OffboardControlMode` | mission_node [60-61](src/mission_manager/mission_manager/mission_node.py#L60-L61) (`publish_offboard_mode`) | PX4 (DDS) | Heartbeat at 50 Hz keeps OFFBOARD alive |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` | mission_node [63-64](src/mission_manager/mission_manager/mission_node.py#L63-L64) (`publish_setpoint`/`publish_velocity`) | PX4 (DDS) | Position or velocity setpoint in **PX4 NED** |
| `/fmu/in/vehicle_command` | `VehicleCommand` | mission_node [66-67](src/mission_manager/mission_manager/mission_node.py#L66-L67) (`send_command`) | PX4 (DDS) | Mode switch (176), arm (400), land (21) |
| `/fmu/in/vehicle_visual_odometry` | `VehicleOdometry` | **vio_bridge** [48-49](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L48-L49) | PX4 EKF2 | **VIO → EKF2 fusion (the position backbone)** |
| `/fmu/in/distance_sensor` | `DistanceSensor` | vio_bridge [50-51](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L50-L51) | PX4 EKF2 | MTF-01 rangefinder altitude |
| `/fmu/out/vehicle_odometry` | `VehicleOdometry` | PX4 (DDS) | mission_node [88-90](src/mission_manager/mission_manager/mission_node.py#L88-L90), octomap [186-187](src/navigation_manager/navigation_manager/octomap_manager.py#L186-L187), obstacle_detector [84-87](src/navigation_manager/navigation_manager/obstacle_detector.py#L84-L87), pose_fusion [79-81](src/navigation_manager/navigation_manager/pose_fusion.py#L79-L81), safety_layer [59-61](src/navigation_manager/navigation_manager/safety_layer.py#L59-L61), depth_filter [73](src/navigation_manager/navigation_manager/depth_filter.py#L73), tf_odom_base [58-63](src/navigation_manager/navigation_manager/tf_odom_base.py#L58-L63), odom_to_path [44-46](src/vio_localization/vio_localization/odom_to_path.py#L44-L46) | **EKF2 fused state — the position EVERY node trusts** |
| `/fmu/out/vehicle_status_v4` | `VehicleStatus` | PX4 (DDS) | mission_node [84-86](src/mission_manager/mission_manager/mission_node.py#L84-L86) | arming_state, nav_state (14=OFFBOARD) |
| `/fmu/out/vehicle_gps_position` | `SensorGps` | PX4 (DDS) | pose_fusion [82-84](src/navigation_manager/navigation_manager/pose_fusion.py#L82-L84) | GPS quality for fusion alpha |

### VIO / SLAM topics

| Topic | Type | Publisher | Subscribers | Data flow |
|-------|------|-----------|-------------|-----------|
| `/oakd/left/image`,`/right/image`,`*/camera_info` | `Image`/`CameraInfo` | gz_bridge_stereo (drone.launch) | stereo_sync [29-38](src/vio_localization/vio_localization/stereo_sync.py#L29-L38) | Raw Gazebo stereo |
| `/oakd/sync/{left,right}/image`,`/camera_info` | `Image`/`CameraInfo` | stereo_sync [51-58](src/vio_localization/vio_localization/stereo_sync.py#L51-L58) | rtabmap_odom (vio.launch remap), OpenVINS | Re-stamped, baseline-corrected stereo |
| `/odom` | `nav_msgs/Odometry` | **rtabmap_odom** (vio.launch) | vio_bridge [41-42](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L41-L42), pose_fusion [85-87](src/navigation_manager/navigation_manager/pose_fusion.py#L85-L87), odom_to_path [29](src/vio_localization/vio_localization/odom_to_path.py#L29) | RTAB-Map stereo VIO (ENU) |
| `/odom_info` | `rtabmap_msgs/OdomInfo` | rtabmap_odom | vio_bridge [39-40](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L39-L40), mission_node [96-98](src/mission_manager/mission_manager/mission_node.py#L96-L98), pose_fusion [88-90](src/navigation_manager/navigation_manager/pose_fusion.py#L88-L90) | VIO quality: `inliers`, `lost` |
| `/openvins/odomimu` | `nav_msgs/Odometry` | OpenVINS (ov_msckf) | mission_node [116-118](src/mission_manager/mission_manager/mission_node.py#L116-L118), odom_to_path [48-50](src/vio_localization/vio_localization/odom_to_path.py#L48-L50) | MSCKF VIO (readiness gate + path viz) |
| `/imu/filtered` | `sensor_msgs/Imu` | imu_filter_madgwick (vio.launch) | depth_filter [74](src/navigation_manager/navigation_manager/depth_filter.py#L74) | Fallback angular rates |
| `/adafruit/imu` | `sensor_msgs/Imu` | gz_bridge_lidars | imu_filter, OpenVINS (imu0) | Raw IMU |
| `/slam/corrected_pose` | `PoseStamped` | **pose_fusion** [92](src/navigation_manager/navigation_manager/pose_fusion.py#L92) | octomap [200-201](src/navigation_manager/navigation_manager/octomap_manager.py#L200-L201) | GPS/VIO fused pose — **octomap only, NOT control** |
| `/pose_fusion/alpha`,`/gps_quality`,`/vio_quality` | `Float32` | pose_fusion [93-95](src/navigation_manager/navigation_manager/pose_fusion.py#L93-L95) | (debug) | Fusion diagnostics |
| `/odom_path`,`/openvins_path` | `nav_msgs/Path` | odom_to_path [28](src/vio_localization/vio_localization/odom_to_path.py#L28),[37](src/vio_localization/vio_localization/odom_to_path.py#L37) | RViz | Trajectory viz |

### Navigation / planning topics

| Topic | Type | Publisher | Subscribers | Data flow |
|-------|------|-----------|-------------|-----------|
| `/current_pose` | `PoseStamped` | **mission_node** [69-70](src/mission_manager/mission_manager/mission_node.py#L69-L70) (in `odom_callback`, World frame) | a_star [112](src/navigation_manager/navigation_manager/a_star_planner.py#L112), path_follower [61](src/navigation_manager/navigation_manager/path_follower.py#L61), waypoint_manager [52-53](src/navigation_manager/navigation_manager/waypoint_manager.py#L52-L53) | **The drone's "where am I" for ALL planning** |
| `/goal_raw` | `PoseStamped` | waypoint_manager [44](src/navigation_manager/navigation_manager/waypoint_manager.py#L44) | mission_node [112-114](src/mission_manager/mission_manager/mission_node.py#L112-L114) | Next waypoint from corridor.yaml |
| `/goal_pose` | `PoseStamped` | mission_node [80-81](src/mission_manager/mission_manager/mission_node.py#L80-L81) (`_publish_goal`) | a_star [113](src/navigation_manager/navigation_manager/a_star_planner.py#L113) | Active goal for planner |
| `/start_navigation` | `Bool` | mission_node [72-73](src/mission_manager/mission_manager/mission_node.py#L72-L73) | waypoint_manager [48-49](src/navigation_manager/navigation_manager/waypoint_manager.py#L48-L49) | Kick off waypoint sequence |
| `/navigation_active` | `Bool` | waypoint_manager [45](src/navigation_manager/navigation_manager/waypoint_manager.py#L45) | mission_node [104-106](src/mission_manager/mission_manager/mission_node.py#L104-L106), a_star [114](src/navigation_manager/navigation_manager/a_star_planner.py#L114), path_follower [63](src/navigation_manager/navigation_manager/path_follower.py#L63) | Mission running / complete flag |
| `/global_path` | `nav_msgs/Path` | a_star [118](src/navigation_manager/navigation_manager/a_star_planner.py#L118) | path_follower [60](src/navigation_manager/navigation_manager/path_follower.py#L60) | 3D smoothed waypoint path (empty=hold) |
| `/desired_velocity` | `TwistStamped` | path_follower [66](src/navigation_manager/navigation_manager/path_follower.py#L66) | safety_layer [55-57](src/navigation_manager/navigation_manager/safety_layer.py#L55-L57) | Velocity toward next waypoint (World) |
| `/safe_velocity` | `TwistStamped` | safety_layer [64-65](src/navigation_manager/navigation_manager/safety_layer.py#L64-L65) | mission_node [92-94](src/mission_manager/mission_manager/mission_node.py#L92-L94) | Velocity after APF obstacle redirection |
| `/goal_reached` | `Bool` | path_follower [67](src/navigation_manager/navigation_manager/path_follower.py#L67) | **(no subscriber)** | Dead-ends — see §5 |
| `/emergency_stop` | `Bool` | (no publisher in these pkgs) | mission_node [100-102](src/mission_manager/mission_manager/mission_node.py#L100-L102), a_star [115](src/navigation_manager/navigation_manager/a_star_planner.py#L115), path_follower [62](src/navigation_manager/navigation_manager/path_follower.py#L62), waypoint_manager [50-51](src/navigation_manager/navigation_manager/waypoint_manager.py#L50-L51) | E-stop bus (no producer found) |
| `/force_update` | `Bool` | mission_node [77-78](src/mission_manager/mission_manager/mission_node.py#L77-L78) | octomap [202-203](src/navigation_manager/navigation_manager/octomap_manager.py#L202-L203) | Lower confirm threshold during stuck recovery |
| `/map_reset` | `Bool` | mission_node [74-75](src/mission_manager/mission_manager/mission_node.py#L74-L75) | octomap [204-205](src/navigation_manager/navigation_manager/octomap_manager.py#L204-L205) | Clear voxel map (never actually published) |

### Mapping / perception topics

| Topic | Type | Publisher | Subscribers | Data flow |
|-------|------|-----------|-------------|-----------|
| `/oakd/depth/image` | `Image` | gz_bridge_depth | depth_filter [72](src/navigation_manager/navigation_manager/depth_filter.py#L72) | Raw depth (320×240) |
| `/pointcloud/camera` | `PointCloud2` | depth_filter [76](src/navigation_manager/navigation_manager/depth_filter.py#L76) | octomap [190-191](src/navigation_manager/navigation_manager/octomap_manager.py#L190-L191) | Depth pixels in **camera frame** |
| `/pointcloud/filtered` | `PointCloud2` | **(no publisher)** | obstacle_detector [88-91](src/navigation_manager/navigation_manager/obstacle_detector.py#L88-L91) | **Dead input — see §5** |
| `/voxel_map` | `PointCloud2` | **octomap** [208](src/navigation_manager/navigation_manager/octomap_manager.py#L208) (+ obstacle_detector [107](src/navigation_manager/navigation_manager/obstacle_detector.py#L107)) | a_star [116](src/navigation_manager/navigation_manager/a_star_planner.py#L116) | **Confirmed obstacle voxels → the ONLY planning map** |
| `/costmap` | `OccupancyGrid` | octomap [207](src/navigation_manager/navigation_manager/octomap_manager.py#L207) (+ obstacle_detector [106](src/navigation_manager/navigation_manager/obstacle_detector.py#L106)) | **(no subscriber)** | 2D costmap — A* ignores it |
| `/obstacle_distances` | `Float32MultiArray` | octomap [209](src/navigation_manager/navigation_manager/octomap_manager.py#L209) (+ obstacle_detector [108](src/navigation_manager/navigation_manager/obstacle_detector.py#L108)) | safety_layer [51-53](src/navigation_manager/navigation_manager/safety_layer.py#L51-L53) | [front,left,right,...] for APF |
| `/global_inflated` | `PointCloud2` | a_star [119](src/navigation_manager/navigation_manager/a_star_planner.py#L119) | RViz | Inflated occupancy viz |
| `/depth_trust_weight` | `Float32` | (no publisher) | octomap [198-199](src/navigation_manager/navigation_manager/octomap_manager.py#L198-L199) | Trust gate (defaults 1.0, never set) |
| `/front_lidar/scan`,`/left_lidar/scan`,`/right_lidar/scan` | `LaserScan` | gz_bridge_lidars | octomap [192-197](src/navigation_manager/navigation_manager/octomap_manager.py#L192-L197), obstacle_detector [92-103](src/navigation_manager/navigation_manager/obstacle_detector.py#L92-L103) | Side/forward range safety |
| `/mtf01/lidar` | `LaserScan` | gz_bridge_lidars | mission_node [108-110](src/mission_manager/mission_manager/mission_node.py#L108-L110), vio_bridge [45-46](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L45-L46), octomap [188-189](src/navigation_manager/navigation_manager/octomap_manager.py#L188-L189), odom_to_path [30](src/vio_localization/vio_localization/odom_to_path.py#L30) | Downward rangefinder = altitude |

---

## 2. NODE DESCRIPTIONS

### 2.1 mission_manager — [mission_node.py](src/mission_manager/mission_manager/mission_node.py)
- **Purpose:** FSM that arms, takes off, builds map, plans, and follows path. Also the **bridge that publishes `/current_pose`** (World frame) from PX4 odometry, and converts `/safe_velocity` (World) into PX4 NED velocity setpoints.
- **Subscriptions:**
  - `/fmu/out/vehicle_status_v4` → `status_callback` [194](src/mission_manager/mission_manager/mission_node.py#L194): sets `armed`, `nav_state`.
  - `/fmu/out/vehicle_odometry` → `odom_callback` [198](src/mission_manager/mission_manager/mission_node.py#L198): stores NED pos, **publishes `/current_pose`** in World.
  - `/safe_velocity` → `safe_velocity_callback` [216](src/mission_manager/mission_manager/mission_node.py#L216).
  - `/odom_info` → `odom_info_callback` [239](src/mission_manager/mission_manager/mission_node.py#L239): VIO inliers gate.
  - `/emergency_stop` → `emergency_callback` [219](src/mission_manager/mission_manager/mission_node.py#L219).
  - `/navigation_active` → `navigation_active_callback` [226](src/mission_manager/mission_manager/mission_node.py#L226): 1 s of inactive in FOLLOW_PATH → LAND.
  - `/mtf01/lidar` → `mtf01_callback` [235](src/mission_manager/mission_manager/mission_node.py#L235): `altitude_m = min(ranges>0.01)`.
  - `/goal_raw` → `goal_raw_callback` [261](src/mission_manager/mission_manager/mission_node.py#L261): republishes to `/goal_pose`.
  - `/openvins/odomimu` → `_openvins_cb` [244](src/mission_manager/mission_manager/mission_node.py#L244): readiness gate (50 stable frames).
- **Publications:** see topic graph. Velocity↔setpoint at 50 Hz from `mission_loop` timer (0.02 s, [186](src/mission_manager/mission_manager/mission_node.py#L186)).
- **Key params:** `SPAWN=(1,3)`, `GOAL=(18,3)`, `TARGET_ALTITUDE=-2.0` NED, `VIO_INLIERS_MIN=15`, `stuck_threshold=200` (4 s), `YAW`: speed thr 0.12, deadband 0.12 rad, max rate 0.30 rad/s.
- **Depends on:** PX4 (DDS), waypoint_manager (`/goal_raw`,`/navigation_active`), safety_layer (`/safe_velocity`), rtabmap (`/odom_info`), OpenVINS.

### 2.2 vio_bridge — [vio_bridge.py](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py)
- **Purpose:** Convert RTAB-Map `/odom` (ENU/FLU) → PX4 `VehicleOdometry` (NED/FRD) on `/fmu/in/vehicle_visual_odometry`, with a quality gate; forward MTF-01 as `DistanceSensor`. **This is the feed that drives EKF2's position estimate.**
- **Subscriptions:** `/odom_info`→`_odom_info_cb` [56](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L56); `/odom`→`_odom_cb` [79](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L79); `/mtf01/lidar`→`_lidar_cb` [166](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L166).
- **Gates (all in `_odom_cb`):**
  1. `if self._odom_lost: return` [81](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L81)
  2. `if self._odom_quality < 15: return` [84](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L84)
  3. Jump gate: reject if `dx>2.0 or dy>2.0` m between samples [95](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L95).
- **`reset_counter`:** incremented on every `0→>0` inlier recovery [70-71](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L70-L71); sent to EKF2 [155](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L155) → tells EKF2 to reset the external-vision reference.
- **Variances hard-coded:** pos 0.01/0.01/0.02, orient 0.01³, vel 0.1³ [143-153](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L143-L153).
- **Launched with `respawn=True`** (csky.launch [31](src/flight_controller_bridge/launch/csky.launch.py#L31)).

### 2.3 a_star_planner — [a_star_planner.py](src/navigation_manager/navigation_manager/a_star_planner.py)
- **Purpose:** The **only** planner. 3D 26-connected A* over the inflated voxel set from `/voxel_map`, then cubic B-spline smoothing. Publishes `/global_path`.
- **Subscriptions:** `/current_pose` [112](src/navigation_manager/navigation_manager/a_star_planner.py#L112), `/goal_pose` [113](src/navigation_manager/navigation_manager/a_star_planner.py#L113), `/navigation_active` [114](src/navigation_manager/navigation_manager/a_star_planner.py#L114), `/emergency_stop` [115](src/navigation_manager/navigation_manager/a_star_planner.py#L115), `/voxel_map` [116](src/navigation_manager/navigation_manager/a_star_planner.py#L116).
- **Decision-maker:** `periodic_check` at 1 Hz [631](src/navigation_manager/navigation_manager/a_star_planner.py#L631) — goal-reached (3D dist < `GOAL_REACHED_DIST=0.8`), goal-changed force replan, plan if `last_path is None`.
- **Key params:** `VOXEL_SIZE=0.1`, `INFLATION_RADIUS=7` voxels (fills full 15³ cube/obstacle [201-204](src/navigation_manager/navigation_manager/a_star_planner.py#L201-L204)), bounds IX[-10,220] IY[-10,80] IZ[5,35] (z 0.5–3.5 m), `MAX_ITER=200000`, spline `s=10`, `KAPPA_MAX=2.0`.

### 2.4 octomap_manager — [octomap_manager.py](src/navigation_manager/navigation_manager/octomap_manager.py)
- **Purpose:** 3D sliding-window evidence map from depth cloud + side LiDAR. Publishes `/voxel_map` (confirmed voxels), `/costmap`, `/obstacle_distances`. **This is the real mapping node** (obstacle_detector is not launched).
- **Subscriptions:** `/fmu/out/vehicle_odometry`→`odom_cb` [220](src/navigation_manager/navigation_manager/octomap_manager.py#L220) (pose history + speed + yaw); `/mtf01/lidar` [255](src/navigation_manager/navigation_manager/octomap_manager.py#L255); `/pointcloud/camera`→`cloud_cb` [360](src/navigation_manager/navigation_manager/octomap_manager.py#L360); 3×side LiDAR; `/depth_trust_weight`; `/slam/corrected_pose`→`slam_pose_cb` [263](src/navigation_manager/navigation_manager/octomap_manager.py#L263); `/force_update`; `/map_reset`.
- **Evidence model:** MARK 8, CONFIRM 150, MAX 600, `CONSISTENCY_FRAMES=30`; confirmed voxels are **position-frozen**; removal only by free-space evidence or 12 m window. Speed-scaled marking [123-131](src/navigation_manager/navigation_manager/octomap_manager.py#L123-L131).
- **Two pose sources, used differently:** cloud projection uses **PX4 odom** via `pose_history` [387](src/navigation_manager/navigation_manager/octomap_manager.py#L387); `drone_x/y` (and jump/loop-closure freeze) come from **`/slam/corrected_pose`** [275-290](src/navigation_manager/navigation_manager/octomap_manager.py#L275-L290). See §5.

### 2.5 path_follower — [path_follower.py](src/navigation_manager/navigation_manager/path_follower.py)
- **Purpose:** Pure velocity generator (no planning). Walks `/global_path`, emits `/desired_velocity`. Publishes `/goal_reached` once.
- **Params:** waypoint radius 0.5 m / z 0.3, goal radius 0.6 / z 0.3, cruise 0.40, max 0.6, max_vz 0.4, vel EMA `alpha=0.4`. Control loop 20 Hz [69](src/navigation_manager/navigation_manager/path_follower.py#L69).
- **Path-change detection** by endpoint+len fingerprint [89-94](src/navigation_manager/navigation_manager/path_follower.py#L89-L94) to avoid index resets.

### 2.6 safety_layer — [safety_layer.py](src/navigation_manager/navigation_manager/safety_layer.py)
- **Purpose:** APF-style repulsion on `/desired_velocity` from `/obstacle_distances`, outputs `/safe_velocity`. Hard front stop < `EMERGENCY_DIST=0.4`. `REPULSE_RANGE=0.5`, `REPULSE_GAIN=0.3`, `LATERAL_STOP=0.4`. 20 Hz watchdog.
- Yaw from `/fmu/out/vehicle_odometry` [88](src/navigation_manager/navigation_manager/safety_layer.py#L88).

### 2.7 waypoint_manager — [waypoint_manager.py](src/navigation_manager/navigation_manager/waypoint_manager.py)
- **Purpose:** Sequences `corridor.yaml` waypoints. Advances when `/current_pose` within `WAYPOINT_RADIUS=1.0` m. Publishes `/goal_raw`, `/navigation_active`. 5 Hz `check_progress`.
- **Waypoints:** (4,3,2)→(8,3,2)→(12,3,1)→(15,3,1)→(18,3,2). On last reached → `/navigation_active=False` → mission lands.

### 2.8 pose_fusion — [pose_fusion.py](src/navigation_manager/navigation_manager/pose_fusion.py)
- **Purpose:** Adaptive GPS(EKF2)/VIO(RTAB) blend → `/slam/corrected_pose` (20 Hz). `alpha=1` pure GPS, `0` pure VIO; EMA-smoothed [173](src/navigation_manager/navigation_manager/pose_fusion.py#L173). VIO quality from `/odom_info` inliers (≥50→1.0 … <5→0). Jump guard >1 m skips [196](src/navigation_manager/navigation_manager/pose_fusion.py#L196). **Output consumed only by octomap.**

### 2.9 depth_filter — [depth_filter.py](src/navigation_manager/navigation_manager/depth_filter.py)
- **Purpose:** Depth image → `/pointcloud/camera` (camera frame). Intrinsics hard-coded FX/FY=161.4, CX=160, CY=120; depth 0.3–6.0 m; subsample 4. **IMU gate drops frames when any body rate > 0.35 rad/s** [101-106](src/navigation_manager/navigation_manager/depth_filter.py#L101-L106).

### 2.10 tf_odom_base — [tf_odom_base.py](src/navigation_manager/navigation_manager/tf_odom_base.py)
- **Purpose:** Broadcast `odom→base_link` TF at 100 Hz from PX4 odometry with NED→ENU −90° yaw correction. Republishes last transform to avoid TF gaps.

### 2.11 stereo_sync — [stereo_sync.py](src/vio_localization/vio_localization/stereo_sync.py)
- **Purpose:** 2-topic ApproximateTimeSync of left/right images (slop 0.10 s), re-stamps both with left stamp, injects baseline `Tx=-fx*0.075` into right `camera_info` [69](src/vio_localization/vio_localization/stereo_sync.py#L69). Republishes `/oakd/sync/*`.

### 2.12 odom_to_path — [odom_to_path.py](src/vio_localization/vio_localization/odom_to_path.py)
- **Purpose:** RViz trajectory viz only. Integrates PX4 velocity + complementary-filters OpenVINS (70/30) into `/openvins_path`; RTAB `/odom`→`/odom_path`. Resets when `lidar_z ≤ 0.3`.

---

## 3. FSM STATES (mission_node.mission_loop, 50 Hz)

| State | Entry condition | Publishes | Exit condition | Next |
|-------|-----------------|-----------|----------------|------|
| **IDLE** (0) | startup | offboard + position setpoint at current NED [424-425](src/mission_manager/mission_manager/mission_node.py#L424-L425) | `offboard_counter ≥ 50` (≈1 s; comment says 2 s) | WAIT_EKF |
| **WAIT_EKF** (1) | from IDLE | offboard + hold setpoint | `wait_counter ≥ 100` (2 s) **AND** `vio_ready` (inliers≥15 or VIO absent) **AND** `openvins_ok`; then sends `DO_SET_MODE→offboard` [462-466](src/mission_manager/mission_manager/mission_node.py#L462-L466) | ARMING |
| **ARMING** (2) | mode cmd sent | offboard + hold setpoint; retries mode every 1 s, arm cmd every 0.5 s | `nav_state==14` **AND** `armed` [494-497](src/mission_manager/mission_manager/mission_node.py#L494-L497) | TAKEOFF |
| **TAKEOFF** (3) | armed+offboard | **velocity** `vz=-0.3` until `altitude_m≥1.8` [500-504](src/mission_manager/mission_manager/mission_node.py#L500-L504) | `altitude_m ≥ 1.8` | BUILD_MAP |
| **BUILD_MAP** (4) | takeoff done | position hold at current NED | `map_build_counter ≥ 100` (2 s); then `_publish_goal()` [521-525](src/mission_manager/mission_manager/mission_node.py#L521-L525) | PLAN_PATH |
| **PLAN_PATH** (5) | map built | hold setpoint + `/goal_pose` + `/start_navigation=True` | `plan_counter ≥ 50` (1 s) **AND** `nav_goal_received` [544](src/mission_manager/mission_manager/mission_node.py#L544) | FOLLOW_PATH |
| **FOLLOW_PATH** (6) | planning done | **velocity** from `/safe_velocity` (World→PX4 swap [560-561](src/mission_manager/mission_manager/mission_node.py#L560-L561)) + smooth yaw; stuck/recovery logic | `/navigation_active=False` for 1 s → LAND; or `/emergency_stop` | LAND / EMERGENCY |
| **EMERGENCY_STOP** (7) | `/emergency_stop=True` | velocity 0 hover | `emergency_counter ≥ 250` (5 s) | LAND |
| **LAND** (8) | mission done / emergency timeout | `VehicleCommand 21` (LAND) + zero setpoint [625-628](src/mission_manager/mission_manager/mission_node.py#L625-L628) | terminal | — |

Stuck recovery inside FOLLOW_PATH: speed<0.02 for >200 ticks (4 s) → hover + `/force_update=True` for 100 ticks (2 s), then resume [564-598](src/mission_manager/mission_manager/mission_node.py#L564-L598).

---

## 4. DATA FLOW PIPELINES

**a. Sensor → position estimate → `/current_pose`**
```
RTAB /odom + side IMU/LiDAR ─▶ PX4 EKF2 ─▶ /fmu/out/vehicle_odometry (NED)
   └▶ mission_node.odom_callback → world_x=East+1, world_y=North+3, z=-Down
      → PUBLISH /current_pose (World, frame_id 'odom')
```
`/current_pose` is **pure EKF2 output** — pose_fusion's `/slam/corrected_pose` is NOT in this path.

**b. `/current_pose` → A* → path → path_follower → velocity → PX4**
```
/current_pose ┐
/goal_pose    ├▶ a_star.periodic_check(1Hz) → 3D A* over /voxel_map → B-spline → /global_path
/voxel_map    ┘
/global_path ─▶ path_follower(20Hz) → unit dir × cruise → /desired_velocity (World)
/desired_velocity + /obstacle_distances ─▶ safety_layer(20Hz) APF → /safe_velocity (World)
/safe_velocity ─▶ mission_node FOLLOW_PATH: px4_vx=world_vy, px4_vy=world_vx, vz=-z,
                  yaw=atan2(world_vx,world_vy) ─▶ /fmu/in/trajectory_setpoint (velocity) ─▶ PX4
```

**c. Depth camera → obstacle detection → costmap → A***
```
/oakd/depth/image ─▶ depth_filter (IMU-gated) → /pointcloud/camera (camera frame)
   ─▶ octomap.cloud_cb: cam→body→NED(via PX4 quat)→World(via pose_history px,py,alt)
      → evidence accumulation → confirmed voxels
   ─▶ /voxel_map (confirmed) ─▶ a_star (the ACTUAL planning input)
   (/costmap published but has NO subscriber)
```

**d. VIO pipeline: cameras → RTAB-Map → vio_bridge → PX4 EKF2**
```
gz stereo ─▶ stereo_sync (re-stamp + baseline Tx) ─▶ /oakd/sync/*
   ─▶ rtabmap_odom (stereo_odometry) ─▶ /odom (ENU) + /odom_info (inliers,lost)
      ─▶ vio_bridge._odom_cb: gates(lost, inliers<15, jump>2m) → ENU→NED, FLU→FRD,
         reset_counter, quality ─▶ /fmu/in/vehicle_visual_odometry ─▶ EKF2 (EKF2_EV_CTRL=9)
MTF-01 /mtf01/lidar ─▶ vio_bridge._lidar_cb ─▶ /fmu/in/distance_sensor ─▶ EKF2 altitude
```

**e. Pose fusion: GPS + VIO → `/slam/corrected_pose` → octomap**
```
/fmu/out/vehicle_odometry (EKF2) ─┐
/fmu/out/vehicle_gps_position     ├▶ pose_fusion._fuse(20Hz): alpha=gpsQ/(gpsQ+vioQ)
/odom (RTAB) + /odom_info         ┘   → blend, jump-guard → /slam/corrected_pose
/slam/corrected_pose ─▶ octomap.slam_pose_cb → sets drone_x/y (costmap origin + jump freeze)
```

---

## 5. KNOWN ISSUES (ranked by relevance to "loses position during flight")

### ⛔ A. VIO dropout + `reset_counter` thrash into EKF2 — PRIME SUSPECT
[vio_bridge.py:70-84](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L70-L84),[155-157](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L155-L157)
- `_odom_cb` **stops publishing entirely** whenever `inliers < 15` or `lost`. During flight along bland corridor walls, fast yaw, or motion blur, RTAB inliers routinely dip below 15 → EKF2 gets **no** external vision and dead-reckons on IMU → drift.
- `Odom/ResetCountdown=2` (vio.launch [119](src/vio_localization/launch/vio.launch.py#L119)) means RTAB **resets its own odometry origin after 2 lost frames**, so `/odom` restarts near zero. On recovery `reset_counter++` is sent to EKF2 [70-71](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L70-L71). Each increment tells EKF2 to re-anchor the EV reference → **step jump in fused position**. Repeated drop/recover cycles = repeated repositioning = exactly "loses position during flight."
- The jump gate raised to **2.0 m** [95](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L95) (comment says "was 1.0") lets fairly large teleports through to EKF2.

### ⛔ B. ENU→NED orientation quaternion conversion is suspect
[vio_bridge.py:118-122](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L118-L122)
- Conversion is a raw component swap `q_ned=(w, y, x, -z)`. Swapping x↔y and negating z is a **reflection (handedness flip)**, not a proper rotation, unless it exactly matches the FLU→FRD + ENU→NED composite. If RTAB's `/odom` orientation is map-ENU (not body-FLU) this yaw is wrong, injecting a biased heading into EKF2 → the position estimate rotates/drifts. Validate against a known-good PX4 EV quaternion helper. The velocity FLU→FRD `( vx, -vy, -vz)` [127-129](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L127-L129) shares the same assumption.

### C. `/current_pose` ignores pose_fusion entirely
[mission_node.py:198-214](src/mission_manager/mission_manager/mission_node.py#L198-L214), [pose_fusion.py:92](src/navigation_manager/navigation_manager/pose_fusion.py#L92)
- All the GPS/VIO adaptive fusion produces `/slam/corrected_pose`, but **navigation control uses raw EKF2** via `/current_pose`. So fusion can't rescue a bad EKF2 estimate for control; it only shifts where octomap *places obstacles*. If EKF2 and corrected_pose diverge, the planner (EKF2 frame) and the map (mixed frames, see D) disagree → drone flies into "free" space that is actually mapped wall, or vice-versa.

### D. octomap mixes two pose sources in one map
[octomap_manager.py:250-290](src/navigation_manager/navigation_manager/octomap_manager.py#L250-L290),[387-411](src/navigation_manager/navigation_manager/octomap_manager.py#L387-L411)
- Cloud projection uses **PX4-odom** pose from `pose_history` (px,py), but `drone_x/y` (costmap origin, FOV free-clearing, jump freeze) comes from **`/slam/corrected_pose`**. When the two frames drift apart, obstacles are stamped at PX4-frame coordinates while free-space logic reasons in corrected-frame → smeared/duplicated walls and bad clearing.
- `slam_pose_cb` rejects jumps >1 m [270](src/navigation_manager/navigation_manager/octomap_manager.py#L270) and **freezes all integration for 20 frames** on loop-closure jumps >0.3 m [279-286](src/navigation_manager/navigation_manager/octomap_manager.py#L279-L286): during VIO corrections the map stops updating, so newly-appearing obstacles are missed.

### E. depth_filter IMU gate blinds mapping during every turn
[depth_filter.py:101-106](src/navigation_manager/navigation_manager/depth_filter.py#L101-L106)
- Any body rate > 0.35 rad/s (≈20°/s) drops the whole depth frame. mission_node's own yaw rate cap is 0.30 rad/s, so commanded turns sit right at the gate → during yawing the costmap goes stale exactly when the drone is reorienting toward obstacles.

### F. `INFLATION_RADIUS=7` voxels with full-cube fill — over-inflation + CPU
[a_star_planner.py:191-204](src/navigation_manager/navigation_manager/a_star_planner.py#L191-L204)
- Each obstacle point fills a 15×15×15 = **3375-cell cube** (0.7 m radius). In a 6 m-wide corridor (60 cells) two opposite walls inflate 0.7 m each → can close the free corridor → A* returns `None` → empty path → drone holds/▶ stuck recovery. Also a Python triple loop per point per `/voxel_map` message is very expensive and can stall the 1 Hz planner.

### G. Duplicate publishers if obstacle_detector is ever enabled
[obstacle_detector.py:106-108](src/navigation_manager/navigation_manager/obstacle_detector.py#L106-L108)
- `obstacle_detector` publishes the **same** `/voxel_map`, `/costmap`, `/obstacle_distances` as octomap and subscribes to the **non-existent** `/pointcloud/filtered` [89](src/navigation_manager/navigation_manager/obstacle_detector.py#L89). It is currently **NOT in navigation.launch.py** (dead code), but if re-added it would race octomap on `/voxel_map` → A* would see two alternating maps. Note it for anyone "fixing" mapping.

### H. Dead/loose wiring
- `/goal_reached` (path_follower [67](src/navigation_manager/navigation_manager/path_follower.py#L67)) has **no subscriber** — mission completion relies solely on waypoint_manager's `/navigation_active`.
- `/emergency_stop` has **no publisher** anywhere in these packages — the whole emergency path and safety_layer's hard-stop never trigger an FSM E-stop unless an external node publishes it.
- `/map_reset` and `/depth_trust_weight` have no publisher → `trust_weight` stays 1.0, `map_reset` never fires.

### I. Goal handshake race in PLAN_PATH
[mission_node.py:261-274](src/mission_manager/mission_manager/mission_node.py#L261-L274),[544](src/mission_manager/mission_manager/mission_node.py#L544); [waypoint_manager.py:63-71](src/navigation_manager/navigation_manager/waypoint_manager.py#L63-L71)
- `nav_goal_received` only set if `/goal_raw` arrives **while** state==PLAN_PATH. `/start_navigation` is BEST_EFFORT depth-10; if waypoint_manager's `start_callback` fires before mission reaches PLAN_PATH, or the Bool is dropped, the first `/goal_raw` can be missed and the FSM waits (it does republish `/goal_pose` every tick, but `nav_goal_received` only flips on a fresh `/goal_raw`).

### J. Timing / counter comments don't match code
- IDLE: `offboard_counter ≥ 50` is ~1 s but comment says "100 ticks = 2 s" [427-428](src/mission_manager/mission_manager/mission_node.py#L427-L428). Minor, but indicates the streaming-before-arm guard is half the intended duration.

### K. QoS durability mismatch risk on PX4 topics — VERIFY
Several subscribers to `/fmu/out/vehicle_odometry` use `DurabilityPolicy.TRANSIENT_LOCAL` (mission_node [45-50](src/mission_manager/mission_manager/mission_node.py#L45-L50), pose_fusion/obstacle_detector/safety_layer/depth_filter/tf_odom_base). If the PX4/Micro-XRCE publisher is `VOLATILE`, a TRANSIENT_LOCAL subscriber is **incompatible and receives nothing**. The system reportedly runs, so the agent likely publishes TRANSIENT_LOCAL — but confirm with `ros2 topic info -v /fmu/out/vehicle_odometry`; a silent QoS drop here would instantly "lose position."

---

## 6. LAUNCH SEQUENCE (csky.launch.py top-level)

`drone.launch` is included at T=0; nested launches add their own relative timers. Global wall-clock:

| Global T | Node(s) | Source |
|----------|---------|--------|
| 0 s | gz_bridge_clock, gz_bridge_lidars, MicroXRCEAgent | drone.launch [99-103](src/flight_controller_bridge/launch/drone.launch.py#L99-L103) |
| 5 s | gz_bridge_depth (alone, 15 s head start) | drone.launch [108-113](src/flight_controller_bridge/launch/drone.launch.py#L108-L113) |
| 10 s | **vio.launch** included → imu_filter, tf_oakd, tf_imu (its T+0) | csky [48-51](src/flight_controller_bridge/launch/csky.launch.py#L48-L51) |
| 15 s | stereo_sync (vio T+5) | vio.launch [189-195](src/vio_localization/launch/vio.launch.py#L189-L195) |
| 20 s | gz_bridge_stereo | drone.launch [117-122](src/flight_controller_bridge/launch/drone.launch.py#L117-L122) |
| 25 s | **vio_bridge** (respawn=True); rtabmap_odom + odom_to_path (vio T+15) | csky [53-56](src/flight_controller_bridge/launch/csky.launch.py#L53-L56), vio.launch [198-205](src/vio_localization/launch/vio.launch.py#L198-L205) |
| 30 s | **navigation.launch** → tf_odom_base, octomap_manager, safety_layer, a_star_planner, path_follower, waypoint_manager, pose_fusion (nav T+0); OpenVINS (vio T+30) | csky [58-61](src/flight_controller_bridge/launch/csky.launch.py#L58-L61), nav.launch [81-90](src/navigation_manager/launch/navigation.launch.py#L81-L90), vio.launch [208-214](src/vio_localization/launch/vio.launch.py#L208-L214) |
| 35 s | depth_filter (nav T+5) | nav.launch [93-98](src/navigation_manager/launch/navigation.launch.py#L93-L98) |
| 45 s | **mission_node** — FSM begins (IDLE) | csky [63-66](src/flight_controller_bridge/launch/csky.launch.py#L63-L66) |

> Note: `navigation.launch.py`'s header comment says it's included at T=58 s, but `csky.launch.py` actually includes it at **T=30 s**. The real timeline is the table above.

Ordering implication: by the time the FSM arms (~T=47 s+), RTAB odometry (T=25 s) and OpenVINS (T=30 s) have had time to initialize; the WAIT_EKF gates (`inliers≥15`, OpenVINS 50 stable frames) enforce this before the offboard/arm commands.

---

## Quick triage checklist for "loses position in flight"
1. `ros2 topic hz /fmu/in/vehicle_visual_odometry` while flying — does it **drop out** when the drone moves/turns? (Issue A)
2. `ros2 topic echo /odom_info` — watch `inliers` cross below 15 and `lost` toggling; correlate with position jumps.
3. Watch `reset_counter` in vio_bridge logs [160](src/flight_controller_bridge/flight_controller_bridge/vio_bridge.py#L160) — every increment is an EKF2 re-anchor.
4. Compare `/current_pose` vs `/slam/corrected_pose` live — divergence confirms Issue C/D.
5. `ros2 topic info -v /fmu/out/vehicle_odometry` — confirm QoS compatibility (Issue K).
6. Check A* logs for `[A*] FAILED` / empty paths during over-inflation (Issue F).
