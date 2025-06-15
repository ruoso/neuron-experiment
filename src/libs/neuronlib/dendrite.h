#ifndef NEURONLIB_DENDRITE_H
#define NEURONLIB_DENDRITE_H

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace neuronlib {

// Address bit field layout:
// Bits 0-2:   Terminal address (if non-zero = terminal, if zero = branch)
// Bits 3-5:   Branch level 1
// Bits 6-8:   Branch level 2  
// Bits 9-11:  Branch level 3
// Bits 12-31: Neuron address (bits 0-11 must be zero for neuron)

constexpr uint32_t TERMINAL_MASK = 0x7;      // Last 3 bits
constexpr uint32_t BRANCH_L1_MASK = 0x38;    // Bits 3-5
constexpr uint32_t BRANCH_L2_MASK = 0x1C0;   // Bits 6-8
constexpr uint32_t BRANCH_L3_MASK = 0xE00;   // Bits 9-11
constexpr uint32_t DENDRITE_MASK = 0xFFF;    // Last 12 bits

HOST_DEVICE inline bool is_terminal_address(uint32_t address) {
    return (address & TERMINAL_MASK) != 0;
}

HOST_DEVICE inline bool is_branch_address(uint32_t address) {
    return (address & TERMINAL_MASK) == 0 && (address & DENDRITE_MASK) != 0;
}

HOST_DEVICE inline bool is_neuron_address(uint32_t address) {
    return (address & DENDRITE_MASK) == 0;
}

HOST_DEVICE inline uint32_t get_terminal_branch(uint32_t terminal_address) {
    return terminal_address & ~TERMINAL_MASK;
}

HOST_DEVICE inline uint32_t get_parent_branch(uint32_t branch_address) {
    if ((branch_address & BRANCH_L1_MASK) != 0) {
        return branch_address & ~BRANCH_L1_MASK;
    } else if ((branch_address & BRANCH_L2_MASK) != 0) {
        return branch_address & ~BRANCH_L2_MASK;
    } else if ((branch_address & BRANCH_L3_MASK) != 0) {
        return branch_address & ~BRANCH_L3_MASK;
    }
    return branch_address;
}

HOST_DEVICE inline uint32_t get_neuron_address(uint32_t dendrite_address) {
    return dendrite_address & ~DENDRITE_MASK;
}

} // namespace neuronlib

#endif // NEURONLIB_DENDRITE_H