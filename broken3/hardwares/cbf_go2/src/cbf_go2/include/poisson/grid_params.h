//good

// grid_params.h
// Grid dimension and resolution constants for the occupancy grid.
// Original source: robot_ws/src/include/poisson/poisson.h
// Only the grid parameters needed for LiDAR occupancy mapping are kept.
#pragma once

#define IMAX 100   // Grid rows (Y dimension)
#define JMAX 100   // Grid columns (X dimension)
#define DS 0.05f   // Cell size in meters (5cm per cell, 100*0.05 = 5m total)
