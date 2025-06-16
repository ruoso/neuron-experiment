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
        if (activation.sensor_index >= MAX_SENSORS) {
            continue;  // Invalid sensor index
        }
        
        const Sensor& sensor = sensor_grid.sensors[activation.sensor_index];
        
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
    
    uint32_t sensor_index = 0;
    for (uint32_t y = 0; y < grid_height && sensor_index < total_sensors; ++y) {
        for (uint32_t x = 0; x < grid_width && sensor_index < total_sensors; ++x) {
            Sensor& sensor = sensor_grid.sensors[sensor_index];
            
            // Set sensor position
            sensor.position.x = min_x + x * spacing_x;
            sensor.position.y = min_y + y * spacing_y;
            sensor.position.z = z_plane;
            
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
                                const Brain& brain,
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
            }
        }
    }
}

} // namespace neuronlib