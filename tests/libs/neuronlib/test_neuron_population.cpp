#include <gtest/gtest.h>
#include "spatial_operations.h"
#include "brain.h"
#include "flow_field.h"
#include "dendrite.h"
#include <cmath>

using namespace neuronlib;

class NeuronPopulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple flow field for testing
        test_flow_field = FlowField3D(0.0f, 0.0f, 0.0f,    // min
                                     10.0f, 10.0f, 10.0f,   // max
                                     2.0f, 2.0f,            // input/output widths
                                     0.1f);                 // resolution
    }
    
    FlowField3D test_flow_field;
    
    float vector_magnitude(const Vec3& v) {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    bool is_within_bounds(const Vec3& pos, const FlowField3D& field) {
        return pos.x >= field.min_x && pos.x <= field.max_x &&
               pos.y >= field.min_y && pos.y <= field.max_y &&
               pos.z >= field.min_z && pos.z <= field.max_z;
    }
};

TEST_F(NeuronPopulationTest, GenerateDendriteTerminals_BasicGeneration) {
    uint32_t neuron_address = 0x1000;  // neuron at address 1 (shifted left by 12 bits)
    Vec3 neuron_position{5.0f, 5.0f, 5.0f};
    
    auto terminals = generate_dendrite_terminals(
        neuron_address, neuron_position, test_flow_field,
        45.0f, 0.1f, 0.5f, 12345);
    
    // Should generate 7^3 * 7 = 2401 terminals
    EXPECT_EQ(terminals.size(), 7 * 7 * 7 * 7);
    
    // Check that all addresses are properly formed
    for (const auto& terminal : terminals) {
        EXPECT_TRUE(is_terminal_address(terminal.terminal_address));
        EXPECT_EQ(get_neuron_address(terminal.terminal_address), neuron_address);
        
        // Terminal should be within reasonable distance
        float distance = vector_magnitude({
            terminal.position.x - neuron_position.x,
            terminal.position.y - neuron_position.y,
            terminal.position.z - neuron_position.z
        });
        EXPECT_GE(distance, 0.1f);
        EXPECT_LE(distance, 0.5f);
    }
}

TEST_F(NeuronPopulationTest, GenerateDendriteTerminals_FlowFieldInfluence) {
    uint32_t neuron_address = 0x1000;
    Vec3 neuron_position{5.0f, 5.0f, 1.0f};  // In input section (positive Z flow)
    
    auto terminals = generate_dendrite_terminals(
        neuron_address, neuron_position, test_flow_field,
        30.0f, 0.2f, 0.3f, 12345);
    
    EXPECT_GT(terminals.size(), 0);
    
    // In input section, flow is in +Z direction, so terminals should generally
    // be biased in that direction (though with random spread)
    float avg_z_offset = 0.0f;
    for (const auto& terminal : terminals) {
        avg_z_offset += (terminal.position.z - neuron_position.z);
    }
    avg_z_offset /= terminals.size();
    
    // Average Z offset should be positive (biased in flow direction)
    EXPECT_GT(avg_z_offset, 0.0f);
}

TEST_F(NeuronPopulationTest, GenerateDendriteTerminals_DeterministicWithSeed) {
    uint32_t neuron_address = 0x1000;
    Vec3 neuron_position{5.0f, 5.0f, 5.0f};
    
    auto terminals1 = generate_dendrite_terminals(
        neuron_address, neuron_position, test_flow_field,
        45.0f, 0.1f, 0.5f, 12345);
    
    auto terminals2 = generate_dendrite_terminals(
        neuron_address, neuron_position, test_flow_field,
        45.0f, 0.1f, 0.5f, 12345);
    
    EXPECT_EQ(terminals1.size(), terminals2.size());
    
    // Should generate identical results with same seed
    for (size_t i = 0; i < terminals1.size(); ++i) {
        EXPECT_EQ(terminals1[i].terminal_address, terminals2[i].terminal_address);
        EXPECT_FLOAT_EQ(terminals1[i].position.x, terminals2[i].position.x);
        EXPECT_FLOAT_EQ(terminals1[i].position.y, terminals2[i].position.y);
        EXPECT_FLOAT_EQ(terminals1[i].position.z, terminals2[i].position.z);
    }
}

TEST_F(NeuronPopulationTest, PopulateNeuronGrid_BasicCreation) {
    auto brain = populate_neuron_grid(test_flow_field, 1.0f, 45.0f, 0.1f, 0.5f, 12345);
    
    ASSERT_NE(brain, nullptr);
    ASSERT_NE(brain->spatial_grid, nullptr);
    ASSERT_NE(brain->spatial_grid->root, nullptr);
    
    // Should have created some neurons
    bool found_neuron = false;
    for (size_t i = 0; i < MAX_NEURONS; ++i) {
        if (brain->neurons[i].threshold == 1.0f) {
            found_neuron = true;
            // Check neuron position is within flow field bounds
            EXPECT_TRUE(is_within_bounds(brain->neurons[i].position, test_flow_field));
            break;
        }
    }
    EXPECT_TRUE(found_neuron);
}

TEST_F(NeuronPopulationTest, PopulateNeuronGrid_NeuronPositioning) {
    auto brain = populate_neuron_grid(test_flow_field, 2.0f, 30.0f, 0.1f, 0.4f, 54321);
    
    // Count actual neurons created
    size_t neuron_count = 0;
    for (size_t i = 0; i < MAX_NEURONS; ++i) {
        if (brain->neurons[i].threshold == 2.0f) {
            neuron_count++;
            
            // Check position is within bounds
            EXPECT_TRUE(is_within_bounds(brain->neurons[i].position, test_flow_field));
            
            // Check that positions are reasonable (not all the same)
            if (i > 0 && brain->neurons[i-1].threshold == 2.0f) {
                Vec3 diff = {
                    brain->neurons[i].position.x - brain->neurons[i-1].position.x,
                    brain->neurons[i].position.y - brain->neurons[i-1].position.y,
                    brain->neurons[i].position.z - brain->neurons[i-1].position.z
                };
                float distance = vector_magnitude(diff);
                // Adjacent neurons should be separated
                if (distance > 0.0f) {
                    EXPECT_GT(distance, 0.01f);  // Some reasonable minimum spacing
                }
            }
        }
    }
    
    EXPECT_GT(neuron_count, 0);
    EXPECT_LT(neuron_count, MAX_NEURONS);  // Shouldn't use all available space
}

TEST_F(NeuronPopulationTest, PopulateNeuronGrid_SpatialGridPopulation) {
    auto brain = populate_neuron_grid(test_flow_field, 1.5f, 60.0f, 0.2f, 0.6f, 98765);
    
    // Test that we can search the spatial grid
    ASSERT_NE(brain->spatial_grid, nullptr);
    
    // Create a frustum that should capture some items
    Frustum3D search_frustum(
        {5.0f, 5.0f, 0.0f},   // apex
        {5.0f, 5.0f, 10.0f},  // base center
        5.0f, 0.0f            // base radius, apex radius
    );
    
    auto results = search_frustum(brain->spatial_grid, search_frustum);
    
    // Should find some items (neurons and/or dendrite terminals)
    EXPECT_GT(results.size(), 0);
    
    // Results should be sorted by distance
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i-1].distance_to_apex, results[i].distance_to_apex);
    }
}

TEST_F(NeuronPopulationTest, PopulateNeuronGrid_AddressConsistency) {
    auto brain = populate_neuron_grid(test_flow_field, 1.0f, 45.0f, 0.1f, 0.5f, 11111);
    
    // Check that neuron addresses are properly formed
    for (size_t i = 0; i < 100; ++i) {  // Check first 100 neurons
        if (brain->neurons[i].threshold == 1.0f) {
            uint32_t expected_address = static_cast<uint32_t>(i) << 12;
            
            // Search spatial grid for this neuron
            Frustum3D small_search(
                brain->neurons[i].position,
                {brain->neurons[i].position.x, brain->neurons[i].position.y, brain->neurons[i].position.z + 0.01f},
                0.01f, 0.0f
            );
            
            auto results = search_frustum(brain->spatial_grid, small_search);
            
            // Should find the neuron itself
            bool found_neuron = false;
            for (const auto& result : results) {
                if (result.item_address == expected_address) {
                    found_neuron = true;
                    break;
                }
            }
            EXPECT_TRUE(found_neuron) << "Neuron " << i << " not found in spatial grid";
        }
    }
}

TEST_F(NeuronPopulationTest, PopulateNeuronGrid_WeightsAndActivationsInitialized) {
    auto brain = populate_neuron_grid(test_flow_field, 1.0f, 45.0f, 0.1f, 0.5f, 22222);
    
    // Check that weights and activations are initialized to zero
    EXPECT_EQ(brain->weights[0], 0.0f);
    EXPECT_EQ(brain->weights[1000], 0.0f);
    EXPECT_EQ(brain->weights[MAX_ADDRESSES - 1], 0.0f);
    
    EXPECT_EQ(brain->last_activations[0], 0);
    EXPECT_EQ(brain->last_activations[1000], 0);
    EXPECT_EQ(brain->last_activations[ACTIVATION_ARRAY_SIZE - 1], 0);
}