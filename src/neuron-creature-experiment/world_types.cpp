#include "world_types.h"
#include <algorithm>

namespace neuron_creature_experiment {

void Tree::update_color_for_state() {
    switch (state.lifecycle_state) {
        case TreeLifecycleState::SEEDLING:
            color = Color(0.4f, 0.8f, 0.4f);  // Light green
            break;
        case TreeLifecycleState::MATURE:
            color = Color(0.2f, 0.6f, 0.2f);  // Dark green
            break;
        case TreeLifecycleState::FRUITING:
            color = Color(0.3f, 0.7f, 0.8f);  // Blue-green (fruiting)
            break;
        case TreeLifecycleState::DORMANT:
            color = Color(0.3f, 0.4f, 0.2f);  // Brown-green
            break;
    }
}

void Fruit::update_color_for_maturity() {
    // Fruits start green and become red as they ripen
    float red_component = maturity;
    float green_component = std::max(0.2f, 1.0f - maturity);
    color = Color(red_component, green_component, 0.1f);
}

} // namespace neuron_creature_experiment