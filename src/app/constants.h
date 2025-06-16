#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <algorithm>

constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr int VIZ_WINDOW_WIDTH = 900;
constexpr int VIZ_WINDOW_HEIGHT = 900;
constexpr int GRID_SIZE = 32;
constexpr int CELL_SIZE = std::min(WINDOW_WIDTH, WINDOW_HEIGHT) / GRID_SIZE;
constexpr int GRID_OFFSET_X = (WINDOW_WIDTH - GRID_SIZE * CELL_SIZE) / 2;
constexpr int GRID_OFFSET_Y = (WINDOW_HEIGHT - GRID_SIZE * CELL_SIZE) / 2;

#endif