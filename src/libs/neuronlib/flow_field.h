#ifndef NEURONLIB_FLOW_FIELD_H
#define NEURONLIB_FLOW_FIELD_H

#include "geometry.h"
#include <memory>
#include <functional>

namespace neuronlib {

struct FlowField3D {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    float input_section_width;
    float output_section_width;
    float resolution;
    Vec3 input_flow_direction;
    Vec3 output_flow_direction;
    bool clockwise_rotation;
    bool normalize_flow;
    
    FlowField3D() 
        : min_x(0), min_y(0), min_z(0)
        , max_x(1), max_y(1), max_z(1)
        , input_section_width(0.2f)
        , output_section_width(0.2f)
        , resolution(0.1f)
        , input_flow_direction{0.0f, 0.0f, 1.0f}
        , output_flow_direction{0.0f, 0.0f, 1.0f}
        , clockwise_rotation(true)
        , normalize_flow(true) {}
        
    FlowField3D(float min_x, float min_y, float min_z, 
                float max_x, float max_y, float max_z,
                float input_width, float output_width, float res)
        : min_x(min_x), min_y(min_y), min_z(min_z)
        , max_x(max_x), max_y(max_y), max_z(max_z)
        , input_section_width(input_width)
        , output_section_width(output_width)
        , resolution(res)
        , input_flow_direction{0.0f, 0.0f, 1.0f}
        , output_flow_direction{0.0f, 0.0f, 1.0f}
        , clockwise_rotation(true)
        , normalize_flow(true) {}
        
    FlowField3D(float min_x, float min_y, float min_z, 
                float max_x, float max_y, float max_z,
                float input_width, float output_width, float res,
                const Vec3& input_dir, const Vec3& output_dir,
                bool clockwise, bool normalize)
        : min_x(min_x), min_y(min_y), min_z(min_z)
        , max_x(max_x), max_y(max_y), max_z(max_z)
        , input_section_width(input_width)
        , output_section_width(output_width)
        , resolution(res)
        , input_flow_direction(input_dir)
        , output_flow_direction(output_dir)
        , clockwise_rotation(clockwise)
        , normalize_flow(normalize) {}
};

using FlowField3DPtr = std::shared_ptr<const FlowField3D>;

using FlowFieldFunction = std::function<Vec3(const Vec3& position)>;

Vec3 evaluate_flow_field(const FlowField3D& field, const Vec3& position);

} // namespace neuronlib

#endif // NEURONLIB_FLOW_FIELD_H