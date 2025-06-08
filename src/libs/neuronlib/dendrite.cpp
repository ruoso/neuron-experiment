#include "dendrite.h"

namespace neuronlib {

bool is_terminal_address(uint32_t address) {
    return (address & TERMINAL_MASK) != 0;
}

bool is_branch_address(uint32_t address) {
    return (address & TERMINAL_MASK) == 0 && (address & DENDRITE_MASK) != 0;
}

bool is_neuron_address(uint32_t address) {
    return (address & DENDRITE_MASK) == 0;
}

uint32_t get_terminal_branch(uint32_t terminal_address) {
    return terminal_address & ~TERMINAL_MASK;
}

uint32_t get_parent_branch(uint32_t branch_address) {
    if ((branch_address & BRANCH_L1_MASK) != 0) {
        return branch_address & ~BRANCH_L1_MASK;
    } else if ((branch_address & BRANCH_L2_MASK) != 0) {
        return branch_address & ~BRANCH_L2_MASK;
    } else if ((branch_address & BRANCH_L3_MASK) != 0) {
        return branch_address & ~BRANCH_L3_MASK;
    }
    return branch_address;
}

uint32_t get_neuron_address(uint32_t dendrite_address) {
    return dendrite_address & ~DENDRITE_MASK;
}

} // namespace neuronlib