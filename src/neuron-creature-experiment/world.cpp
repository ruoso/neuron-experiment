#include "world.h"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace neuron_creature_experiment {

World::World(const WorldConfig& config) 
    : config_(config), current_tick_(0), next_tree_id_(1), next_fruit_id_(1),
      rng_(std::random_device{}()), uniform_dist_(0.0f, 1.0f) {
    trees_.reserve(config_.max_trees);
    fruits_.reserve(config_.max_fruits);
}

void World::step_simulation(uint32_t tick) {
    current_tick_ = tick;
    
    update_trees(tick);
    update_fruits(tick);
    update_creature(tick);
}


float World::distance(const Vec2& pos1, const Vec2& pos2) const {
    // No wrapping - use standard Euclidean distance
    float dx = pos1.x - pos2.x;
    float dy = pos1.y - pos2.y;
    
    return std::sqrt(dx * dx + dy * dy);
}

void World::add_tree(const Vec2& position) {
    if (trees_.size() >= config_.max_trees) return;
    
    trees_.emplace_back(position, next_tree_id());
    trees_.back().update_color_for_state();
}

void World::remove_tree(uint32_t tree_id) {
    trees_.erase(std::remove_if(trees_.begin(), trees_.end(),
        [tree_id](const Tree& tree) { return tree.tree_id == tree_id; }),
        trees_.end());
}

void World::add_fruit(const Vec2& position, uint32_t parent_tree_id) {
    if (fruits_.size() >= config_.max_fruits) return;
    
    fruits_.emplace_back(position, next_fruit_id(), parent_tree_id);
    fruits_.back().update_color_for_maturity();
}

void World::remove_fruit(uint32_t fruit_id) {
    fruits_.erase(std::remove_if(fruits_.begin(), fruits_.end(),
        [fruit_id](const Fruit& fruit) { return fruit.fruit_id == fruit_id; }),
        fruits_.end());
}

void World::set_creature_position(const Vec2& position) {
    creature_state_.position = position;
}

void World::set_creature_orientation(float orientation) {
    creature_state_.orientation = orientation;
}

void World::apply_motor_force(float left_force, float right_force) {
    float force_diff = right_force - left_force;
    float total_force = (left_force + right_force) * 0.5f;
    
    creature_state_.angular_velocity += force_diff * config_.simulation_dt;
    
    float forward_x = std::cos(creature_state_.orientation) * total_force;
    float forward_y = std::sin(creature_state_.orientation) * total_force;
    
    creature_state_.velocity.x += forward_x * config_.simulation_dt;
    creature_state_.velocity.y += forward_y * config_.simulation_dt;
}

bool World::consume_if_in_range() {
    for (auto& fruit : fruits_) {
        if (!fruit.available) continue;
        
        float dist = distance(creature_state_.position, fruit.position);
        
        if (dist <= config_.creature_eat_radius) {
            creature_state_.hunger = std::max(0.0f, creature_state_.hunger - fruit.satiation_value);
            creature_state_.energy = std::min(1.0f, creature_state_.energy + fruit.satiation_value * 0.1f);
            fruit.available = false;
            SPDLOG_INFO("Fruit consumed! Position: ({:.2f}, {:.2f}), Satiation: {:.2f}", 
                        fruit.position.x, fruit.position.y, fruit.satiation_value);
            return true;
        }
    }
    
    return false;
}


std::vector<VisionSample> World::get_visible_objects(const Vec2& position, float angle, float fov) const {
    std::vector<VisionSample> samples(NUM_VISION_STRIPS);
    
    float strip_angle = fov / static_cast<float>(NUM_VISION_STRIPS);
    float start_angle = angle - fov * 0.5f;
    
    // Initialize all samples
    for (int i = 0; i < NUM_VISION_STRIPS; ++i) {
        samples[i] = VisionSample();
    }
    
    // Debug: Log vision request
    static uint32_t vision_call_count = 0;
    vision_call_count++;
    if (vision_call_count % 60 == 0) { // Log every 60 calls (once per second at 60fps)
        SPDLOG_DEBUG("Vision system called {} times. Position: ({:.2f}, {:.2f}), Angle: {:.2f}, FOV: {:.2f}, Trees: {}, Fruits: {}", 
                     vision_call_count, position.x, position.y, angle, fov, trees_.size(), fruits_.size());
        
        // Log some nearby objects for debugging
        int logged_fruits = 0;
        for (const auto& fruit : fruits_) {
            if (!fruit.available) continue;
            float obj_distance = distance(position, fruit.position);
            if (obj_distance < 20.0f && logged_fruits < 3) {
                SPDLOG_DEBUG("  Nearby fruit: pos=({:.2f}, {:.2f}), dist={:.2f}, maturity={:.2f}", 
                             fruit.position.x, fruit.position.y, obj_distance, fruit.maturity);
                logged_fruits++;
            }
        }
        
        int logged_trees = 0;
        for (const auto& tree : trees_) {
            float tree_distance = distance(position, tree.position);
            if (tree_distance < 20.0f && logged_trees < 3) {
                SPDLOG_DEBUG("  Nearby tree: pos=({:.2f}, {:.2f}), dist={:.2f}, state={}", 
                             tree.position.x, tree.position.y, tree_distance, static_cast<int>(tree.state.lifecycle_state));
                logged_trees++;
            }
        }
    }
    
    // Process fruits
    process_objects_for_vision<Fruit>(
        fruits_, samples, position, start_angle, strip_angle,
        [](const Fruit& fruit) { return fruit.available; },
        [](const Fruit& fruit, float distance) {
            return std::max(0.1f, fruit.maturity); // Minimum visibility even when immature
        }
    );
    
    // Process trees
    process_objects_for_vision<Tree>(
        trees_, samples, position, start_angle, strip_angle,
        [](const Tree& tree) { return true; },
        [](const Tree& tree, float distance) {
            return 1.0f; // Constant intensity, angular size determines contribution
        }
    );
    
    // Debug: Check if we got any vision signals
    int active_strips = 0;
    float total_intensity = 0.0f;
    for (const auto& sample : samples) {
        if (sample.total_intensity > 0.0f) {
            active_strips++;
            total_intensity += sample.total_intensity;
        }
    }
    
    if (vision_call_count % 60 == 0 && active_strips == 0) {
        spdlog::warn("No vision signals detected! Active strips: {}, Total intensity: {:.3f}", active_strips, total_intensity);
    }
    
    return samples;
}

CreatureState World::get_creature_state() const {
    return creature_state_;
}

void World::update_trees(uint32_t tick) {
    for (auto& tree : trees_) {
        tree.state.age += config_.simulation_dt;
        tree.state.state_timer += config_.simulation_dt;
        
        TreeLifecycleState old_state = tree.state.lifecycle_state;
        
        switch (tree.state.lifecycle_state) {
            case TreeLifecycleState::SEEDLING:
                if (tree.state.state_timer > 10.0f) {
                    tree.state.lifecycle_state = TreeLifecycleState::MATURE;
                    tree.state.state_timer = 0.0f;
                }
                break;
                
            case TreeLifecycleState::MATURE:
                if (tree.state.state_timer > 15.0f) {
                    tree.state.lifecycle_state = TreeLifecycleState::FRUITING;
                    tree.state.state_timer = 0.0f;
                    spawn_fruits_from_tree(tree);
                }
                break;
                
            case TreeLifecycleState::FRUITING:
                if (tree.state.state_timer > 20.0f) {
                    tree.state.lifecycle_state = TreeLifecycleState::DORMANT;
                    tree.state.state_timer = 0.0f;
                }
                break;
                
            case TreeLifecycleState::DORMANT:
                if (tree.state.state_timer > 10.0f) {
                    // Tree becomes old and dies - mark for removal
                    tree.tree_id = 0; // Use tree_id = 0 to mark for removal
                }
                break;
        }
        
        // Update color based on age (always, since it's linear with age)
        tree.update_color_for_state();
    }
    
    // Remove trees that have completed their lifecycle
    size_t trees_before = trees_.size();
    trees_.erase(std::remove_if(trees_.begin(), trees_.end(),
        [](const Tree& tree) { return tree.tree_id == 0; }),
        trees_.end());
    size_t trees_after = trees_.size();
    
    if (trees_before != trees_after) {
        SPDLOG_INFO("Removed {} old trees. Total trees: {} -> {}", 
                    trees_before - trees_after, trees_before, trees_after);
    }
    
    // Spawn new trees to maintain stable population
    // Target: maintain around 15-20 trees by spawning more frequently around creature
    if (trees_.size() < config_.max_trees && uniform_dist_(rng_) < 0.05f) {
        // Spawn within 40 units but at least 12 units away from creature
        // (12 units = creature eat radius 5 + max fruit spawn distance 5 + safety buffer 2)
        Vec2 spawn_pos;
        int attempts = 0;
        do {
            float x = creature_state_.position.x + (uniform_dist_(rng_) - 0.5f) * 80.0f; // -40 to +40
            float y = creature_state_.position.y + (uniform_dist_(rng_) - 0.5f) * 80.0f; // -40 to +40
            spawn_pos = Vec2(x, y);
            attempts++;
        } while (distance(creature_state_.position, spawn_pos) < 12.0f && attempts < 10);
        
        if (attempts < 10) {
            add_tree(spawn_pos);
            SPDLOG_INFO("Spawned new tree at ({:.2f}, {:.2f}) near creature ({:.2f}, {:.2f}), distance: {:.2f}. Total trees: {}", 
                       spawn_pos.x, spawn_pos.y, creature_state_.position.x, creature_state_.position.y, 
                       distance(creature_state_.position, spawn_pos), trees_.size());
        }
    }
}

void World::update_fruits(uint32_t tick) {
    for (auto& fruit : fruits_) {
        if (!fruit.available) continue;
        
        fruit.maturity = std::min(1.5f, fruit.maturity + config_.simulation_dt * 0.02f);
        
        // Mark fruit for removal if it goes past maximum maturity (overripe)
        if (fruit.maturity > 1.0f) {
            fruit.available = false; // Mark as unavailable (will be removed)
        } else {
            fruit.update_satiation(config_.fruit_max_satiation);
            fruit.update_color_for_maturity();
        }
    }
    
    size_t fruits_before = fruits_.size();
    fruits_.erase(std::remove_if(fruits_.begin(), fruits_.end(),
        [](const Fruit& fruit) { return !fruit.available; }),
        fruits_.end());
    size_t fruits_after = fruits_.size();
    
    if (fruits_before != fruits_after) {
        SPDLOG_INFO("Removed {} fruits (consumed/overripe). Total fruits: {} -> {}", 
                    fruits_before - fruits_after, fruits_before, fruits_after);
    }
}

void World::update_creature(uint32_t tick) {
    creature_state_.position = creature_state_.position + creature_state_.velocity * config_.simulation_dt;
    
    creature_state_.orientation += creature_state_.angular_velocity * config_.simulation_dt;
    
    
    creature_state_.hunger += config_.hunger_increase_rate * config_.simulation_dt;
    creature_state_.hunger = std::min(1.0f, creature_state_.hunger);
}

void World::spawn_fruits_from_tree(const Tree& tree) {
    uint32_t num_fruits = 3 + static_cast<uint32_t>(uniform_dist_(rng_) * 5);
    
    for (uint32_t i = 0; i < num_fruits; ++i) {
        float angle = uniform_dist_(rng_) * 2.0f * M_PI;
        float radius = 2.0f + uniform_dist_(rng_) * 3.0f;
        
        Vec2 fruit_pos;
        fruit_pos.x = tree.position.x + std::cos(angle) * radius;
        fruit_pos.y = tree.position.y + std::sin(angle) * radius;
        
        add_fruit(fruit_pos, tree.tree_id);
        // Set initial maturity for newly spawned fruits
        if (!fruits_.empty()) {
            fruits_.back().maturity = 0.1f + uniform_dist_(rng_) * 0.3f; // 0.1 to 0.4 initial maturity
            fruits_.back().update_color_for_maturity();
        }
    }
}

bool World::is_fruit_visible(const Fruit& fruit, const Vec2& eye_pos, float angle, float fov) const {
    Vec2 to_fruit = fruit.position - eye_pos;
    
    float fruit_angle = std::atan2(to_fruit.y, to_fruit.x);
    float angle_diff = std::abs(fruit_angle - angle);
    
    if (angle_diff > M_PI) angle_diff = 2.0f * M_PI - angle_diff;
    
    return angle_diff <= fov * 0.5f;
}

uint32_t World::next_tree_id() {
    return next_tree_id_++;
}

uint32_t World::next_fruit_id() {
    return next_fruit_id_++;
}

} // namespace neuron_creature_experiment