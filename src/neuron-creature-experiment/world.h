#ifndef NEURON_CREATURE_EXPERIMENT_WORLD_H
#define NEURON_CREATURE_EXPERIMENT_WORLD_H

#include "world_types.h"
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace neuron_creature_experiment {

constexpr int NUM_VISION_STRIPS = 64;

class World {
public:
    World(const WorldConfig& config);
    ~World() = default;

    void step_simulation(uint32_t tick);
    
    float distance(const Vec2& pos1, const Vec2& pos2) const;
    
    void add_tree(const Vec2& position);
    void remove_tree(uint32_t tree_id);
    
    void add_fruit(const Vec2& position, uint32_t parent_tree_id);
    void remove_fruit(uint32_t fruit_id);
    
    void set_creature_position(const Vec2& position);
    void set_creature_orientation(float orientation);
    void apply_motor_force(float left_force, float right_force);
    bool consume_if_in_range();
    
    std::vector<VisionSample> get_visible_objects(const Vec2& position, float angle, float fov) const;
    CreatureState get_creature_state() const;
    
    const std::vector<Tree>& get_trees() const { return trees_; }
    const std::vector<Fruit>& get_fruits() const { return fruits_; }
    
    uint32_t get_current_tick() const { return current_tick_; }

private:
    void update_trees(uint32_t tick);
    void update_fruits(uint32_t tick);
    void update_creature(uint32_t tick);
    
    void spawn_fruits_from_tree(const Tree& tree);
    bool is_fruit_visible(const Fruit& fruit, const Vec2& eye_pos, float angle, float fov) const;
    
    struct ObjectForVision {
        Vec2 position;
        float radius;
        Color color;
        float distance;
        float intensity;
        float start_angle;
        float end_angle;
    };
    
    template<typename ObjectType, typename Container>
    void process_objects_for_vision(
        const Container& objects,
        std::vector<VisionSample>& samples,
        const Vec2& position,
        float start_angle,
        float strip_angle,
        std::function<bool(const ObjectType&)> is_visible,
        std::function<float(const ObjectType&, float)> get_intensity
    ) const {
        // First, collect all visible objects with their angular ranges
        std::vector<ObjectForVision> visible_objects;
        
        static uint32_t debug_call_count = 0;
        debug_call_count++;
        bool should_debug = (debug_call_count % 60 == 0);
        
        for (const auto& obj : objects) {
            if (!is_visible(obj)) continue;
            
            Vec2 to_obj = obj.position - position;
            float obj_center_angle = std::atan2(to_obj.y, to_obj.x);
            float distance = this->distance(position, obj.position);
            
            // Normalize object angle to be relative to creature's facing direction
            float angle_diff = obj_center_angle - start_angle;
            while (angle_diff > M_PI) angle_diff -= 2.0f * M_PI;
            while (angle_diff < -M_PI) angle_diff += 2.0f * M_PI;
            obj_center_angle = start_angle + angle_diff;
            
            // Calculate angular extent of object
            float angular_radius = std::atan(obj.radius / std::max(1.0f, distance));
            float obj_start_angle = obj_center_angle - angular_radius;
            float obj_end_angle = obj_center_angle + angular_radius;
            
            float base_intensity = get_intensity(obj, distance);
            
            if (should_debug && visible_objects.size() < 3) {
                SPDLOG_DEBUG("    Object at ({:.2f}, {:.2f}): center_angle={:.2f}, start_angle={:.2f}, end_angle={:.2f}, intensity={:.2f}",
                             obj.position.x, obj.position.y, obj_center_angle, obj_start_angle, obj_end_angle, base_intensity);
                SPDLOG_DEBUG("    Vision range: {:.2f} to {:.2f}, strip_angle={:.2f}",
                             start_angle, start_angle + NUM_VISION_STRIPS * strip_angle, strip_angle);
            }
            
            visible_objects.push_back({
                obj.position, obj.radius, obj.color, distance, base_intensity,
                obj_start_angle, obj_end_angle
            });
        }
        
        // Sort objects by distance (closest first for proper occlusion)
        std::sort(visible_objects.begin(), visible_objects.end(),
                  [](const ObjectForVision& a, const ObjectForVision& b) {
                      return a.distance < b.distance;
                  });
        
        // Track occlusion per strip (how much of each strip is already blocked)
        std::vector<float> strip_occlusion(NUM_VISION_STRIPS, 0.0f);
        
        // Process objects from closest to farthest
        for (const auto& obj : visible_objects) {
            // Check overlap with each vision strip
            for (int strip = 0; strip < NUM_VISION_STRIPS; ++strip) {
                float strip_start_angle = start_angle + strip * strip_angle;
                float strip_end_angle = start_angle + (strip + 1) * strip_angle;
                
                // Calculate overlap between object's angular range and this strip
                float overlap_start = std::max(obj.start_angle, strip_start_angle);
                float overlap_end = std::min(obj.end_angle, strip_end_angle);
                float overlap = std::max(0.0f, overlap_end - overlap_start);
                
                if (overlap > 0.0f) {
                    // Calculate what fraction of the strip is covered
                    float coverage_ratio = overlap / strip_angle;
                    
                    // Apply occlusion - only the unoccluded portion contributes
                    float visible_ratio = std::max(0.0f, 1.0f - strip_occlusion[strip]);
                    float effective_coverage = coverage_ratio * visible_ratio;
                    
                    if (effective_coverage > 0.0f) {
                        float contribution = obj.intensity * effective_coverage;
                        
                        // Add to blended color
                        samples[strip].blended_color.r += obj.color.r * contribution;
                        samples[strip].blended_color.g += obj.color.g * contribution;
                        samples[strip].blended_color.b += obj.color.b * contribution;
                        samples[strip].total_intensity += contribution;
                        
                        // Update closest distance
                        if (samples[strip].distance == 0.0f || obj.distance < samples[strip].distance) {
                            samples[strip].distance = obj.distance;
                        }
                    }
                    
                    // Update occlusion for this strip
                    strip_occlusion[strip] = std::min(1.0f, strip_occlusion[strip] + coverage_ratio);
                }
            }
        }
    }
    
    uint32_t next_tree_id();
    uint32_t next_fruit_id();

    WorldConfig config_;
    std::vector<Tree> trees_;
    std::vector<Fruit> fruits_;
    CreatureState creature_state_;
    
    uint32_t current_tick_;
    uint32_t next_tree_id_;
    uint32_t next_fruit_id_;
    
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;
};

} // namespace neuron_creature_experiment

#endif // NEURON_CREATURE_EXPERIMENT_WORLD_H