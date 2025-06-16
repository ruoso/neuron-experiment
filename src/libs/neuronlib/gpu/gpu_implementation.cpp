#include "gpu_interface.h"
#include "../neuron.h"
#include "gpu_kernel_source.h"
#include <CL/cl.h>
#include <memory>
#include <cstdio>
#include <vector>

namespace neuronlib {
namespace gpu {

// Forward declaration of kernel
std::string load_kernel_source();

// OpenCL memory management class
class GpuMemoryManager {
private:
    // OpenCL context and resources
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel kernel_;
    
    // Device buffers
    cl_mem d_input_;
    cl_mem d_output_;
    
public:
    GpuMemoryManager() : context_(nullptr), queue_(nullptr), program_(nullptr), 
                        kernel_(nullptr), d_input_(nullptr), d_output_(nullptr) {}
    
    ~GpuMemoryManager() {
        cleanup();
    }
    
    bool initialize() {
        cl_int err;
        
        // Get platform
        cl_uint num_platforms;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to query OpenCL platforms: %d\n", err);
            return false;
        }
        
        printf("GPU Info: Found %d OpenCL platforms\n", num_platforms);
        if (num_platforms == 0) {
            printf("GPU Init Error: No OpenCL platforms found. Please install OpenCL drivers.\n");
            return false;
        }
        
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to get OpenCL platform: %d\n", err);
            return false;
        }
        
        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to get OpenCL device: %d\n", err);
            return false;
        } else {
            printf("GPU Info: Using GPU device\n");
        }
        
        // Create context
        context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to create OpenCL context: %d\n", err);
            return false;
        }
        
        // Create command queue
        queue_ = clCreateCommandQueueWithProperties(context_, device, nullptr, &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to create command queue: %d\n", err);
            return false;
        }
        
        // Load and compile kernel
        std::string kernel_source = load_kernel_source();
        const char* source_ptr = kernel_source.c_str();
        size_t source_size = kernel_source.length();
        
        program_ = clCreateProgramWithSource(context_, 1, &source_ptr, &source_size, &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to create program: %d\n", err);
            return false;
        }
        
        err = clBuildProgram(program_, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to build program: %d\n", err);
            
            // Get build log
            size_t log_size;
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            printf("Build log: %s\n", log.data());
            return false;
        }
        
        // Create kernel
        kernel_ = clCreateKernel(program_, "process_activations_kernel", &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to create kernel: %d\n", err);
            return false;
        }
        
        // Allocate device memory
        d_input_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, sizeof(GpuActivationBatch), nullptr, &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to allocate input buffer: %d\n", err);
            return false;
        }
        
        d_output_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(GpuProcessingResults), nullptr, &err);
        if (err != CL_SUCCESS) {
            printf("GPU Init Error: Failed to allocate output buffer: %d\n", err);
            return false;
        }
        
        return true;
    }
    
    bool process_batch(const GpuActivationBatch& input, GpuProcessingResults& output) {
        cl_int err;
        
        // Copy input to device
        err = clEnqueueWriteBuffer(queue_, d_input_, CL_TRUE, 0, sizeof(GpuActivationBatch), &input, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to write input buffer: %d\n", err);
            return false;
        }
        
        // Clear output structure
        output.clear();
        err = clEnqueueWriteBuffer(queue_, d_output_, CL_TRUE, 0, sizeof(GpuProcessingResults), &output, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to write output buffer: %d\n", err);
            return false;
        }
        
        // Set kernel arguments
        err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_input_);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to set kernel arg 0: %d\n", err);
            return false;
        }
        
        err = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_output_);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to set kernel arg 1: %d\n", err);
            return false;
        }
        
        // Execute kernel
        size_t global_work_size = input.target_count;
        size_t local_work_size = 256; // Workgroup size
        
        // Round up to multiple of local work size
        if (global_work_size % local_work_size != 0) {
            global_work_size = ((global_work_size + local_work_size - 1) / local_work_size) * local_work_size;
        }
        
        err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to execute kernel: %d\n", err);
            return false;
        }
        
        // Wait for completion
        err = clFinish(queue_);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to wait for kernel completion: %d\n", err);
            return false;
        }
        
        // Read results back
        err = clEnqueueReadBuffer(queue_, d_output_, CL_TRUE, 0, sizeof(GpuProcessingResults), &output, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            printf("GPU Error: Failed to read output buffer: %d\n", err);
            return false;
        }
        
        return true;
    }
    
    void cleanup() {
        if (d_input_) { clReleaseMemObject(d_input_); d_input_ = nullptr; }
        if (d_output_) { clReleaseMemObject(d_output_); d_output_ = nullptr; }
        if (kernel_) { clReleaseKernel(kernel_); kernel_ = nullptr; }
        if (program_) { clReleaseProgram(program_); program_ = nullptr; }
        if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
    }
};

// Global GPU memory manager instance
static std::unique_ptr<GpuMemoryManager> g_gpu_manager = nullptr;

// Load kernel source from generated header
std::string load_kernel_source() {
    return std::string(gpu_kernel_source);
}

// Initialize GPU system
bool initialize_gpu() {
    if (g_gpu_manager) {
        return true;  // Already initialized
    }
    
    // Create memory manager
    g_gpu_manager = std::make_unique<GpuMemoryManager>();
    if (!g_gpu_manager->initialize()) {
        printf("GPU Init Error: Failed to initialize GPU memory manager\n");
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