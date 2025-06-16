#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <algorithm>

enum class CellSource {
    NONE = 0,
    USER = 1,
    ACTUATOR = 2
};

struct GridCell {
    float user_intensity;      // 0.0 to 1.0
    float actuator_intensity;  // 0.0 to 1.0
    CellSource last_source;
    bool activation_sent;      // Track if activation was already sent
    
    GridCell() : user_intensity(0.0f), actuator_intensity(0.0f), last_source(CellSource::NONE), activation_sent(false) {}
    
    float get_total_intensity() const {
        return std::max(user_intensity, actuator_intensity);
    }
    
    uint8_t get_gray_value() const {
        return static_cast<uint8_t>(get_total_intensity() * 255);
    }
};

struct RippleEffect {
    int center_x, center_y;
    float current_radius;
    float max_radius;
    CellSource source_type;
    uint32_t start_time;
    
    RippleEffect(int x, int y, CellSource source, uint32_t time) 
        : center_x(x), center_y(y), current_radius(0.0f), max_radius(4.0f), 
          source_type(source), start_time(time) {}
};

struct IsometricPoint {
    float x, y;
};

#endif