#include "gpu_interface.h"
#include "../neuron.h"
#include <cuda_runtime.h>
#include <memory>

namespace neuronlib {
namespace gpu {

// Forward declaration of kernel
__global__ void process_activations_kernel(
    const GpuActivationBatch* input,
    GpuProcessingResults* output
);

// GPU memory management class
class GpuMemoryManager {
private:
    // Device pointers
    GpuActivationBatch* d_input_;
    GpuProcessingResults* d_output_;
    
public:
    GpuMemoryManager() : d_input_(nullptr), d_output_(nullptr) {}
    
    ~GpuMemoryManager() {
        cleanup();
    }
    
    bool initialize() {
        // Allocate GPU memory for input/output structures
        cudaError_t err;
        
        err = cudaMalloc(&d_input_, sizeof(GpuActivationBatch));
        if (err != cudaSuccess) {
            return false;
        }
        
        err = cudaMalloc(&d_output_, sizeof(GpuProcessingResults));
        if (err != cudaSuccess) {
            return false;
        }
        
        return true;
    }
    
    
    bool process_batch(const GpuActivationBatch& input, GpuProcessingResults& output) {
        cudaError_t err;
        
        // Copy input to GPU
        err = cudaMemcpy(d_input_, &input, sizeof(GpuActivationBatch), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return false;
        }
        
        // Clear output structure
        output.clear();
        err = cudaMemcpy(d_output_, &output, sizeof(GpuProcessingResults), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            return false;
        }
        
        // Launch kernel
        int block_size = 256;  // Threads per block
        int grid_size = (input.target_count + block_size - 1) / block_size;  // Blocks in grid
        
        
        process_activations_kernel<<<grid_size, block_size>>>(
            d_input_, d_output_
        );
        
        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return false;
        }
        
        // Wait for kernel to complete
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return false;
        }
        
        // Copy result back to host
        err = cudaMemcpy(&output, d_output_, sizeof(GpuProcessingResults), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            return false;
        }
        
        
        return true;
    }
    
    void cleanup() {
        if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
        if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    }
};

// Global GPU memory manager instance
static std::unique_ptr<GpuMemoryManager> g_gpu_manager = nullptr;

// Initialize GPU system
bool initialize_gpu() {
    if (g_gpu_manager) {
        return true;  // Already initialized
    }
    
    // Check for CUDA-capable devices
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return false;
    }
    
    
    // Create memory manager
    g_gpu_manager = std::make_unique<GpuMemoryManager>();
    if (!g_gpu_manager->initialize()) {
        g_gpu_manager.reset();
        return false;
    }
    
    return true;
}

// Cleanup GPU system
void cleanup_gpu() {
    g_gpu_manager.reset();
}

// Main GPU processing interface function
bool process_activations_gpu(const GpuActivationBatch& input, 
                            GpuProcessingResults& output) {
    if (!g_gpu_manager) {
        return false;
    }
    
    // Process the batch
    return g_gpu_manager->process_batch(input, output);
}

} // namespace gpu
} // namespace neuronlib