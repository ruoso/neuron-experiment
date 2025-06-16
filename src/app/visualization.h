#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "types.h"
#include "geometry.h"
#include <cstdint>
#include <cmath>

struct Color {
    uint8_t r, g, b;
};

using namespace neuronlib;

IsometricPoint project_to_isometric(const Vec3& point);
Color depth_to_color(float z_position);

#endif