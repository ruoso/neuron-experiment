#include "flow_field.h"
#include <cmath>

namespace neuronlib {

Vec3 evaluate_flow_field(const FlowField3D& field, const Vec3& position) {
    float input_section_end = field.min_z + field.input_section_width;
    float output_section_start = field.max_z - field.output_section_width;
    
    // Input section: configurable flow direction
    if (position.z <= input_section_end) {
        return field.input_flow_direction;
    }
    
    // Output section: configurable flow direction
    if (position.z >= output_section_start) {
        return field.output_flow_direction;
    }
    
    // Middle section: circular flow
    // Calculate center of the middle section
    float center_x = (field.min_x + field.max_x) * 0.5f;
    float center_y = (field.min_y + field.max_y) * 0.5f;
    
    // Relative position from center
    float rel_x = position.x - center_x;
    float rel_y = position.y - center_y;
    
    // Circular flow: tangent vector to the circle
    float flow_x, flow_y;
    if (field.clockwise_rotation) {
        // Clockwise rotation: (-y, x)
        flow_x = -rel_y;
        flow_y = rel_x;
    } else {
        // Counter-clockwise rotation: (y, -x)
        flow_x = rel_y;
        flow_y = -rel_x;
    }
    
    // Optionally normalize the flow vector
    if (field.normalize_flow) {
        float magnitude = std::sqrt(flow_x * flow_x + flow_y * flow_y);
        if (magnitude > 0.0f) {
            flow_x /= magnitude;
            flow_y /= magnitude;
        }
    }
    
    // Pure circular flow in middle section
    return {flow_x, flow_y, 0.0f};
}

} // namespace neuronlib