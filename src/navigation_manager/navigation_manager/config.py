# ==============================================================
# CSKy Drone — Shared Configuration
# Single source of truth for all nodes
# ==============================================================

# Spawn offset (PX4 local origin → World frame)
SPAWN_X = 1.0   # metres
SPAWN_Y = 3.0   # metres

# Corridor dimensions
CORRIDOR_LENGTH = 20.0   # metres (X axis)
CORRIDOR_WIDTH  = 6.0    # metres (Y axis)
CORRIDOR_HEIGHT = 3.0    # metres (Z axis)

# 3D voxel grid
RESOLUTION = 0.10        # metres per cell
GRID_W = int(CORRIDOR_LENGTH / RESOLUTION)   # 200
GRID_H = int(CORRIDOR_WIDTH  / RESOLUTION)   # 60
GRID_Z = int(CORRIDOR_HEIGHT / RESOLUTION)   # 30

# Grid world origin
ORIGIN_X = 0.0
ORIGIN_Y = 0.0
ORIGIN_Z = 0.0

# Flight parameters
TARGET_ALTITUDE = 2.0    # metres (world frame)
GOAL_X = 18.0            # metres
GOAL_Y = 3.0             # metres
