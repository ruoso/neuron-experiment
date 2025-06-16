#include "visualization.h"
#include "constants.h"
#include "spatial_operations.h"
#include <cmath>
#include <algorithm>

IsometricPoint project_to_isometric(const Vec3& point) {
    // Side view projection: rotate by -20° around Y for slight angle, then 20° around X for top-down tilt
    float cos_y = 0.940f;  // cos(-20°)
    float sin_y = 0.342f;  // sin(-20°) 
    float cos_x = 0.940f;  // cos(20°)
    float sin_x = 0.342f;  // sin(20°)
    
    // Scale and center the coordinates
    float scale = 250.0f;
    float center_x = VIZ_WINDOW_WIDTH / 2.0f;
    float center_y = VIZ_WINDOW_HEIGHT / 2.0f;
    
    // Apply Y rotation first (horizontal viewing angle) - flip the rotation
    float x1 = point.x * cos_y + point.z * sin_y;  // Changed sign to flip view
    float y1 = point.y;
    float z1 = -point.x * sin_y + point.z * cos_y;  // Changed sign to flip view
    
    // Then apply X rotation (vertical viewing angle)
    float x2 = x1;
    float y2 = y1 * cos_x - z1 * sin_x;
    
    return {center_x + x2 * scale, center_y - y2 * scale};
}

Color depth_to_color(float z_position) {
    // Convert Z position (-1.0 to +1.0) to hue (blue to red)
    // Sensors at z=-1.0 (front) = blue (240°)
    // Actuators at z=+1.0 (back) = red (0°)
    float normalized_depth = (z_position + 1.0f) / 2.0f;  // 0.0 to 1.0
    float hue = 240.0f * (1.0f - normalized_depth);  // 240° to 0°
    
    // HSV to RGB conversion with full saturation and value
    float saturation = 1.0f;
    float value = 1.0f;
    
    float c = value * saturation;
    float h_prime = hue / 60.0f;
    float x = c * (1.0f - std::abs(std::fmod(h_prime, 2.0f) - 1.0f));
    
    float r, g, b;
    if (h_prime >= 0.0f && h_prime < 1.0f) {
        r = c; g = x; b = 0.0f;
    } else if (h_prime >= 1.0f && h_prime < 2.0f) {
        r = x; g = c; b = 0.0f;
    } else if (h_prime >= 2.0f && h_prime < 3.0f) {
        r = 0.0f; g = c; b = x;
    } else if (h_prime >= 3.0f && h_prime < 4.0f) {
        r = 0.0f; g = x; b = c;
    } else if (h_prime >= 4.0f && h_prime < 5.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }
    
    return {
        static_cast<uint8_t>((r) * 255),
        static_cast<uint8_t>((g) * 255),
        static_cast<uint8_t>((b) * 255)
    };
}