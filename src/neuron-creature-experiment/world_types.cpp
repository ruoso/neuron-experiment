#include "world_types.h"
#include <algorithm>

namespace neuron_creature_experiment {

void Tree::update_color_for_state() {
    // Calculate overall tree maturity based on total age
    // Trees live for 10+15+20+10 = 55 seconds total
    float total_lifetime = 55.0f;
    float maturity = std::min(1.0f, state.age / total_lifetime);
    
    // Update radius based on lifecycle state
    switch (state.lifecycle_state) {
        case TreeLifecycleState::SEEDLING:
            // Radius grows from 0 to full size (3.0) during seedling phase
            radius = (state.state_timer / 10.0f) * 3.0f;
            break;
        case TreeLifecycleState::MATURE:
        case TreeLifecycleState::FRUITING:
        case TreeLifecycleState::DORMANT:
            radius = 3.0f; // Full size
            break;
    }
    
    // Linear interpolation: light green (0.6, 0.9, 0.4) -> dark green (0.1, 0.4, 0.1)
    float light_r = 0.6f, light_g = 0.9f, light_b = 0.4f;
    float dark_r = 0.1f, dark_g = 0.4f, dark_b = 0.1f;
    
    color.r = light_r + (dark_r - light_r) * maturity;
    color.g = light_g + (dark_g - light_g) * maturity;
    color.b = light_b + (dark_b - light_b) * maturity;
}

void Fruit::update_color_for_maturity() {
    // Linear interpolation from yellow (young) to brown (mature)
    // Yellow: (0.9, 0.8, 0.2) -> Brown: (0.6, 0.3, 0.1)
    float yellow_r = 0.9f, yellow_g = 0.8f, yellow_b = 0.2f;
    float brown_r = 0.6f, brown_g = 0.3f, brown_b = 0.1f;
    
    color.r = yellow_r + (brown_r - yellow_r) * maturity;
    color.g = yellow_g + (brown_g - yellow_g) * maturity;
    color.b = yellow_b + (brown_b - yellow_b) * maturity;
}

} // namespace neuron_creature_experiment