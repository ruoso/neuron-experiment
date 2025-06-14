#include "flow_field.h"
#include <cmath>

namespace neuronlib {

Vec3 evaluate_flow_field(const FlowField3D& field, const Vec3& position) {
    float input_section_end = field.min_z + field.input_section_width;
    float output_section_start = field.max_z - field.output_section_width;
    
    // Input section: positive Z direction
    if (position.z <= input_section_end) {
        return {0.0f, 0.0f, 1.0f};
    }
    
    // Output section: positive Z direction
    if (position.z >= output_section_start) {
        return {0.0f, 0.0f, 1.0f};
    }
    
    // Middle section: circular flow with output bias
    // Calculate center of the middle section
    float center_x = (field.min_x + field.max_x) * 0.5f;
    float center_y = (field.min_y + field.max_y) * 0.5f;
    
    // Relative position from center
    float rel_x = position.x - center_x;
    float rel_y = position.y - center_y;
    
    // Circular flow: tangent vector to the circle
    // For clockwise rotation: (-y, x)
    // For counter-clockwise rotation: (y, -x)
    float flow_x = -rel_y;  // clockwise
    float flow_y = rel_x;
    
    // Normalize the flow vector
    float magnitude = std::sqrt(flow_x * flow_x + flow_y * flow_y);
    if (magnitude > 0.0f) {
        flow_x /= magnitude;
        flow_y /= magnitude;
    }
    
    // Progressive bias towards output: stronger Z component as we approach output
    float middle_section_length = output_section_start - input_section_end;
    float progress = (position.z - input_section_end) / middle_section_length;
    float z_bias = 0.1f + (0.9f * progress);  // Ranges from 0.1 to 0.5
    
    return {flow_x, flow_y, z_bias};
}

} // namespace neuronlib