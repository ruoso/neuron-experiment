#ifndef NEURONLIB_GPU_CONVERTER_H
#define NEURONLIB_GPU_CONVERTER_H

#include "gpu_interface.h"
#include "../activation.h"
#include <unordered_map>
#include <vector>

namespace neuronlib {

struct Brain;  // Forward declaration

namespace gpu {

// Converter class to transform CPU data structures to/from GPU format
class GpuConverter {
public:
    // Convert from CPU pending_messages to GPU batch format
    // Returns number of batches created (may be > 1 if data doesn't fit in one batch)
    static size_t convert_to_gpu_batches(
        const std::unordered_map<uint32_t, std::vector<Activation>>& pending_messages,
        uint32_t current_timestamp,
        uint32_t timing_window,
        const Brain& brain,
        std::vector<GpuActivationBatch>& output_batches
    );
    
    
    // Helper function to estimate how many batches will be needed
    static size_t estimate_batch_count(
        const std::unordered_map<uint32_t, std::vector<Activation>>& pending_messages,
        uint32_t current_timestamp,
        uint32_t timing_window
    );
    
private:
    // Helper to count valid activations within timing window
    static size_t count_valid_activations(
        const std::vector<Activation>& activations,
        uint32_t current_timestamp,
        uint32_t timing_window
    );
};

} // namespace gpu
} // namespace neuronlib

#endif // NEURONLIB_GPU_CONVERTER_H