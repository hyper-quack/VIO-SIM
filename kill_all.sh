#!/usr/bin/env bash
# Kills the entire CSKy stack in reverse startup order.
# Safe to run multiple times — pkill exits 1 if no match, errors suppressed.
set -euo pipefail

kill_proc() {
    local name="$1"
    if pkill -SIGTERM -f "$name" 2>/dev/null; then
        echo "[kill] SIGTERM → $name"
    fi
}

force_kill() {
    local name="$1"
    if pkill -SIGKILL -f "$name" 2>/dev/null; then
        echo "[kill] SIGKILL → $name (forced)"
    fi
}

echo "════════════════════════════════════════════════════"
echo "  CSKy stack teardown — $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════"

# ── Step 1: mission_node (top-level FSM — disarm first) ──────────────────────
kill_proc "mission_node"

# ── Step 2: navigation stack ─────────────────────────────────────────────────
kill_proc "depth_filter"
kill_proc "waypoint_manager"
kill_proc "path_follower"
kill_proc "a_star_planner"
kill_proc "safety_layer"
kill_proc "obstacle_detector"
pkill -9 px4
pkill -9 gz
pkill -9 ruby

# ── Step 3: vio_bridge (stops sending odometry to PX4) ───────────────────────
kill_proc "vio_bridge"

# ── Step 4: rtabmap stack ─────────────────────────────────────────────────────
kill_proc "rtabmap"           # rtabmap_slam executable
kill_proc "stereo_odometry"   # rtabmap_odom executable
kill_proc "stereo_sync"
kill_proc "imu_filter_madgwick_node"
kill_proc "static_transform_publisher"

# ── Step 5: gz bridges ────────────────────────────────────────────────────────
# Kill by bridge name so we don't accidentally kill unrelated parameter_bridge
# processes. All three share the 'parameter_bridge' executable, so we match
# the ROS node name flag that appears in /proc/<pid>/cmdline.
kill_proc "gz_bridge_stereo"
kill_proc "gz_bridge_depth"
kill_proc "gz_bridge_lidars"
# Fallback: kill any remaining parameter_bridge if the above missed them
kill_proc "parameter_bridge"

# ── Step 6: MicroXRCEAgent ────────────────────────────────────────────────────
kill_proc "MicroXRCEAgent"

# ── Step 7: give processes 3 seconds to exit cleanly, then force-kill ─────────
sleep 3

force_kill "mission_node"
force_kill "depth_filter"
force_kill "waypoint_manager"
force_kill "path_follower"
force_kill "a_star_planner"
force_kill "safety_layer"
force_kill "obstacle_detector"
force_kill "vio_bridge"
force_kill "rtabmap"
force_kill "stereo_odometry"
force_kill "stereo_sync"
force_kill "imu_filter_madgwick_node"
force_kill "parameter_bridge"
force_kill "MicroXRCEAgent"

# ── Step 8: restart ROS 2 daemon to clear stale topic registrations ───────────
echo "[kill] Restarting ROS 2 daemon..."
ros2 daemon stop  2>/dev/null || true
sleep 1
ros2 daemon start 2>/dev/null || true

echo "════════════════════════════════════════════════════"
echo "  Done. Verify with: ros2 node list"
echo "════════════════════════════════════════════════════"
