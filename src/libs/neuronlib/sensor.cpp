#include "sensor.h"
#include "brain.h"
#include "spatial_operations.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace neuronlib {

std::vector<TargetedActivation> process_sensor_activations(
    const SensorGrid& sensor_grid,
    const std::vector<SensorActivation>& activations,
    uint32_t timestamp) {
    
    std::vector<TargetedActivation> result;
    
    for (const auto& activation : activations) {
        // Find all sensors with this tag
        auto tag_range = sensor_grid.tag_to_index.equal_range(activation.sensor_tag);
        
        for (auto it = tag_range.first; it != tag_range.second; ++it) {
            uint32_t sensor_index = it->second;
            if (sensor_index >= MAX_SENSORS) {
                continue;  // Invalid sensor index
            }
            
            const Sensor& sensor = sensor_grid.sensors[sensor_index];
            
            // Check each mode bit in the bitmap
            for (uint32_t mode_index = 0; mode_index < NUM_ACTIVATION_MODES; ++mode_index) {
                if (activation.mode_bitmap & (1 << mode_index)) {
                    // This mode is activated
                    const ActivationMode& mode = sensor.modes[mode_index];
                    
                    // Send activation to all dendrites in this mode
                    for (size_t dendrite_index = 0; dendrite_index < MAX_DENDRITES_PER_MODE; ++dendrite_index) {
                        uint32_t target_dendrite = mode.target_dendrites[dendrite_index];
                        if (target_dendrite != 0) {
                            // send the message to the branch, giving the terminal address as source
                            result.emplace_back(get_terminal_branch(target_dendrite), Activation(activation.value, timestamp, target_dendrite));
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

void populate_sensor_grid(SensorGrid& sensor_grid,
                         uint32_t grid_width, uint32_t grid_height,
                         float min_x, float min_y, float z_plane,
                         float max_x, float max_y) {
    
    sensor_grid.grid_width = grid_width;
    sensor_grid.grid_height = grid_height;
    sensor_grid.min_x = min_x;
    sensor_grid.min_y = min_y;
    sensor_grid.min_z = z_plane;
    sensor_grid.max_x = max_x;
    sensor_grid.max_y = max_y;
    sensor_grid.max_z = z_plane;
    
    uint32_t total_sensors = grid_width * grid_height;
    if (total_sensors > MAX_SENSORS) {
        total_sensors = MAX_SENSORS;
    }
    
    float spacing_x = (max_x - min_x) / (grid_width - 1);
    float spacing_y = (max_y - min_y) / (grid_height - 1);
    
    // Clear tag mapping
    sensor_grid.tag_to_index.clear();
    
    uint32_t sensor_index = 0;
    for (uint32_t y = 0; y < grid_height && sensor_index < total_sensors; ++y) {
        for (uint32_t x = 0; x < grid_width && sensor_index < total_sensors; ++x) {
            Sensor& sensor = sensor_grid.sensors[sensor_index];
            
            // Set sensor position
            sensor.position.x = min_x + x * spacing_x;
            sensor.position.y = min_y + y * spacing_y;
            sensor.position.z = z_plane;
            
            // Set sensor tag to match index for compatibility
            sensor.sensor_tag = static_cast<uint16_t>(sensor_index);
            sensor_grid.tag_to_index[sensor.sensor_tag] = sensor_index;
            
            // Initialize all modes with empty dendrite lists
            // (Dendrite connections will be set up by external functions)
            for (uint32_t mode_index = 0; mode_index < NUM_ACTIVATION_MODES; ++mode_index) {
                for (size_t dendrite_index = 0; dendrite_index < MAX_DENDRITES_PER_MODE; ++dendrite_index) {
                    sensor.modes[mode_index].target_dendrites[dendrite_index] = 0;
                }
            }
            
            sensor_index++;
        }
    }
}

void assign_dendrites_to_sensors(SensorGrid& sensor_grid,
                                Brain& brain,
                                float connection_radius,
                                uint32_t random_seed) {
    
    std::mt19937 rng(random_seed);
    
    uint32_t total_sensors = sensor_grid.grid_width * sensor_grid.grid_height;
    if (total_sensors > MAX_SENSORS) {
        total_sensors = MAX_SENSORS;
    }
    
    for (uint32_t sensor_index = 0; sensor_index < total_sensors; ++sensor_index) {
        Sensor& sensor = sensor_grid.sensors[sensor_index];
        
        // Create a frustum to search for nearby dendrites
        Vec3 search_center = sensor.position;
        Vec3 search_end = {sensor.position.x, sensor.position.y, sensor.position.z + connection_radius};
        Frustum3D search_volume(search_center, search_end, connection_radius, 0.0f);
        
        // Find all dendrite terminals within connection radius
        auto search_results = search_frustum(brain.spatial_grid, search_volume);
        
        // Filter to only dendrite terminals (not neurons)
        std::vector<uint32_t> nearby_dendrites;
        for (const auto& result : search_results) {
            if (is_terminal_address(result.item_address)) {
                // check that weight is zero, since we only want unconnected dendrites
                if (brain.weights[result.item_address] != 0.0f) {
                    continue; // Skip connected dendrites
                }
                nearby_dendrites.push_back(result.item_address);
            }
        }
        
        // Shuffle the dendrites for random assignment
        std::shuffle(nearby_dendrites.begin(), nearby_dendrites.end(), rng);
        
        // Assign dendrites to modes
        size_t dendrite_index = 0;
        for (uint32_t mode_index = 0; mode_index < NUM_ACTIVATION_MODES; ++mode_index) {
            ActivationMode& mode = sensor.modes[mode_index];
            
            // Clear existing assignments
            for (size_t i = 0; i < MAX_DENDRITES_PER_MODE; ++i) {
                mode.target_dendrites[i] = 0;
            }
            
            // Assign dendrites to this mode
            for (size_t mode_dendrite_index = 0; 
                 mode_dendrite_index < MAX_DENDRITES_PER_MODE && dendrite_index < nearby_dendrites.size(); 
                 ++mode_dendrite_index, ++dendrite_index) {
                
                mode.target_dendrites[mode_dendrite_index] = nearby_dendrites[dendrite_index];
                // now assign a weight to the dendrite, using a normal distribution
                float weight = std::normal_distribution<float>(0.5f, 0.167f)(rng);
                brain.weights[nearby_dendrites[dendrite_index]] = weight;
            }
        }
    }
}

void populate_sensor_grid_with_positions(SensorGrid& sensor_grid,
                                        const std::vector<SensorPosition>& sensor_positions) {
    // Clear existing sensor grid
    sensor_grid.grid_width = 0;
    sensor_grid.grid_height = 0;
    sensor_grid.tag_to_index.clear();
    
    // Calculate bounding box of all sensor positions
    if (sensor_positions.empty()) {
        sensor_grid.min_x = sensor_grid.min_y = sensor_grid.min_z = 0.0f;
        sensor_grid.max_x = sensor_grid.max_y = sensor_grid.max_z = 0.0f;
        return;
    }
    
    // Initialize bounding box with first position
    sensor_grid.min_x = sensor_grid.max_x = sensor_positions[0].position.x;
    sensor_grid.min_y = sensor_grid.max_y = sensor_positions[0].position.y;
    sensor_grid.min_z = sensor_grid.max_z = sensor_positions[0].position.z;
    
    // Expand bounding box to include all positions
    for (const auto& sensor_pos : sensor_positions) {
        sensor_grid.min_x = std::min(sensor_grid.min_x, sensor_pos.position.x);
        sensor_grid.max_x = std::max(sensor_grid.max_x, sensor_pos.position.x);
        sensor_grid.min_y = std::min(sensor_grid.min_y, sensor_pos.position.y);
        sensor_grid.max_y = std::max(sensor_grid.max_y, sensor_pos.position.y);
        sensor_grid.min_z = std::min(sensor_grid.min_z, sensor_pos.position.z);
        sensor_grid.max_z = std::max(sensor_grid.max_z, sensor_pos.position.z);
    }
    
    // Populate sensors with custom positions and tags
    size_t sensor_count = std::min(sensor_positions.size(), static_cast<size_t>(MAX_SENSORS));
    for (size_t i = 0; i < sensor_count; ++i) {
        sensor_grid.sensors[i].position = sensor_positions[i].position;
        sensor_grid.sensors[i].sensor_tag = sensor_positions[i].sensor_tag;
        sensor_grid.tag_to_index[sensor_positions[i].sensor_tag] = static_cast<uint32_t>(i);
        
        // Initialize all modes with empty dendrite lists
        for (uint32_t mode_index = 0; mode_index < NUM_ACTIVATION_MODES; ++mode_index) {
            for (size_t dendrite_index = 0; dendrite_index < MAX_DENDRITES_PER_MODE; ++dendrite_index) {
                sensor_grid.sensors[i].modes[mode_index].target_dendrites[dendrite_index] = 0;
            }
        }
    }
    
    // Set grid dimensions (for compatibility, although not used in this mode)
    sensor_grid.grid_width = static_cast<uint32_t>(sensor_count);
    sensor_grid.grid_height = 1;
}

} // namespace neuronlib