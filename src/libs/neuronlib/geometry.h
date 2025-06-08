#ifndef NEURONLIB_GEOMETRY_H
#define NEURONLIB_GEOMETRY_H

namespace neuronlib {

struct Vec3 {
    float x, y, z;
};

struct Cone3D {
    Vec3 apex;
    Vec3 base_center;
    float base_radius;
    float apex_radius;
    
    Cone3D() : apex{0, 0, 0}, base_center{0, 0, 0}, base_radius(0.0f), apex_radius(0.0f) {}
    Cone3D(const Vec3& apex_pos, const Vec3& base_pos, float base_r, float apex_r = 0.0f)
        : apex(apex_pos), base_center(base_pos), base_radius(base_r), apex_radius(apex_r) {}
};

using Frustum3D = Cone3D;

} // namespace neuronlib

#endif // NEURONLIB_GEOMETRY_H