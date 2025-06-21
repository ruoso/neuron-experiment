#include "genetic_algorithm.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

namespace genetic_algorithm {

GeneticAlgorithm::GeneticAlgorithm(const GAParameters& params) 
    : params_(params), rng_(params.random_seed) {
    
    // Create output directory
    std::filesystem::create_directories(params_.output_directory);
    
    std::cout << "Genetic Algorithm initialized:" << std::endl;
    std::cout << "  Population size: " << params_.population_size << std::endl;
    std::cout << "  Generations: " << params_.num_generations << std::endl;
    std::cout << "  Mutation rate: " << params_.mutation_rate << std::endl;
    std::cout << "  Crossover rate: " << params_.crossover_rate << std::endl;
    std::cout << "  Elite size: " << params_.elite_size << std::endl;
    std::cout << "  Output directory: " << params_.output_directory << std::endl;
}

void GeneticAlgorithm::run() {
    std::cout << "Starting genetic algorithm evolution..." << std::endl;
    
    initialize_population();
    
    for (uint32_t generation = 0; generation < params_.num_generations; ++generation) {
        std::cout << "\n=== Generation " << generation + 1 << " / " << params_.num_generations << " ===" << std::endl;
        
        evaluate_population();
        save_generation_results(generation);
        
        // Find best individual
        auto best_it = std::max_element(population_.begin(), population_.end(),
            [](const Individual& a, const Individual& b) {
                return a.fitness < b.fitness;
            });
        
        std::cout << "Best fitness: " << best_it->fitness 
                  << " (survived: " << best_it->ticks_survived 
                  << ", moved: " << best_it->distance_moved 
                  << ", ate: " << best_it->fruits_eaten << ")" << std::endl;
        
        if (generation < params_.num_generations - 1) {
            select_and_reproduce();
        } else {
            save_best_individual(*best_it);
        }
    }
    
    std::cout << "\nGenetic algorithm completed!" << std::endl;
}

void GeneticAlgorithm::initialize_population() {
    std::cout << "Initializing random population..." << std::endl;
    
    population_.clear();
    population_.reserve(params_.population_size);
    
    for (uint32_t i = 0; i < params_.population_size; ++i) {
        std::string layout = generate_random_layout_encoding();
        population_.emplace_back(layout);
    }
    
    std::cout << "Generated " << population_.size() << " random individuals" << std::endl;
}

std::string GeneticAlgorithm::generate_random_layout_encoding() {
    // Same logic as in creature_experiment.cpp but with different random state
    constexpr int NUM_VISION_SENSORS = 192;
    constexpr int total_positions = NUM_VISION_SENSORS + 8 * 4 + 2; // vision + 4 motor types (8 each) + hunger + satiation
    constexpr int values_per_position = 3; // x, y, z
    constexpr int total_values = total_positions * values_per_position;
    
    // Generate random unsigned short values
    std::uniform_int_distribution<uint16_t> dis(0, 65535);
    
    std::vector<uint16_t> values;
    values.reserve(total_values);
    for (int i = 0; i < total_values; ++i) {
        values.push_back(dis(rng_));
    }
    
    // Convert to base64
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(values.data());
    size_t byte_count = values.size() * sizeof(uint16_t);
    
    std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    
    for (size_t i = 0; i < byte_count; i += 3) {
        uint32_t group = 0;
        int group_size = 0;
        
        for (int j = 0; j < 3 && (i + j) < byte_count; ++j) {
            group = (group << 8) | bytes[i + j];
            group_size++;
        }
        
        // Pad to 24 bits
        group <<= (3 - group_size) * 8;
        
        // Extract 6-bit chunks
        for (int j = 3; j >= 0; --j) {
            if (j < 4 - ((3 - group_size) * 4 / 3)) {
                result += base64_chars[(group >> (j * 6)) & 0x3F];
            } else {
                result += '=';
            }
        }
    }
    
    return result;
}

std::string GeneticAlgorithm::mutate_layout(const std::string& layout, float mutation_rate) {
    // Decode base64 to bytes, mutate some bytes, re-encode
    std::string mutated = layout;
    std::uniform_real_distribution<float> mut_prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> char_dist(0, 255);
    
    for (size_t i = 0; i < mutated.length(); ++i) {
        if (mut_prob(rng_) < mutation_rate) {
            // Replace with random base64 character
            std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::uniform_int_distribution<size_t> char_idx(0, base64_chars.length() - 1);
            mutated[i] = base64_chars[char_idx(rng_)];
        }
    }
    
    return mutated;
}

std::string GeneticAlgorithm::crossover_layouts(const std::string& parent1, const std::string& parent2) {
    // Simple single-point crossover
    size_t min_length = std::min(parent1.length(), parent2.length());
    std::uniform_int_distribution<size_t> crossover_point(1, min_length - 1);
    size_t point = crossover_point(rng_);
    
    return parent1.substr(0, point) + parent2.substr(point);
}

void GeneticAlgorithm::evaluate_population() {
    std::cout << "Evaluating population fitness..." << std::endl;
    
    for (size_t i = 0; i < population_.size(); ++i) {
        std::cout << "  Individual " << (i + 1) << "/" << population_.size() << "... ";
        std::cout.flush();
        
        float fitness = evaluate_individual(population_[i]);
        population_[i].fitness = fitness;
        
        std::cout << "fitness = " << fitness << std::endl;
    }
}

float GeneticAlgorithm::evaluate_individual(Individual& individual) {
    // Create unique output filename for this individual
    std::stringstream ss;
    ss << params_.output_directory << "/individual_" << std::hash<std::string>{}(individual.layout_encoding) << ".txt";
    std::string output_file = ss.str();
    
    // Run simulation
    if (!run_simulation(individual.layout_encoding, output_file)) {
        return 0.0f; // Failed simulation gets zero fitness
    }
    
    // Parse results
    std::ifstream file(output_file);
    if (!file.is_open()) {
        return 0.0f;
    }
    
    std::string line;
    if (std::getline(file, line)) {
        individual.ticks_survived = std::stoul(line);
    }
    if (std::getline(file, line)) {
        individual.distance_moved = std::stof(line);
    }
    if (std::getline(file, line)) {
        individual.fruits_eaten = std::stoul(line);
    }
    
    file.close();
    
    return calculate_fitness(individual.ticks_survived, individual.distance_moved, individual.fruits_eaten);
}

bool GeneticAlgorithm::run_simulation(const std::string& layout_encoding, const std::string& output_file) {
    // Fork and exec the creature experiment with headless mode
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process - run the simulation
        std::string executable = "build/bin/creature-experiment";
        execl(executable.c_str(), "creature-experiment", layout_encoding.c_str(), nullptr);
        
        // If we reach here, exec failed
        std::cerr << "Failed to execute simulation" << std::endl;
        exit(1);
    } else if (pid > 0) {
        // Parent process - wait for completion
        int status;
        waitpid(pid, &status, 0);
        
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    } else {
        // Fork failed
        std::cerr << "Failed to fork simulation process" << std::endl;
        return false;
    }
}

float GeneticAlgorithm::calculate_fitness(uint32_t ticks_survived, float distance_moved, uint32_t fruits_eaten) {
    // Multi-objective fitness function
    // Prioritize survival time, but also reward movement and eating
    float fitness = 0.0f;
    
    // Base survival fitness (logarithmic to avoid extreme values)
    fitness += std::log(1.0f + ticks_survived) * 10.0f;
    
    // Movement bonus (exploration)
    fitness += distance_moved * 0.1f;
    
    // Eating bonus (survival efficiency)
    fitness += fruits_eaten * 50.0f;
    
    return fitness;
}

void GeneticAlgorithm::select_and_reproduce() {
    std::cout << "Selecting and reproducing..." << std::endl;
    
    // Sort population by fitness
    std::sort(population_.begin(), population_.end(),
        [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });
    
    std::vector<Individual> new_population;
    new_population.reserve(params_.population_size);
    
    // Keep elite individuals
    for (uint32_t i = 0; i < std::min(params_.elite_size, static_cast<uint32_t>(population_.size())); ++i) {
        new_population.push_back(population_[i]);
    }
    
    // Fill rest with tournament selection and reproduction
    while (new_population.size() < params_.population_size) {
        auto parents = tournament_selection(2, 3);
        
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::string child_layout;
        
        if (prob(rng_) < params_.crossover_rate) {
            // Crossover
            child_layout = crossover_layouts(parents[0].layout_encoding, parents[1].layout_encoding);
        } else {
            // Just take one parent
            child_layout = parents[0].layout_encoding;
        }
        
        // Mutate
        child_layout = mutate_layout(child_layout, params_.mutation_rate);
        
        new_population.emplace_back(child_layout);
    }
    
    population_ = std::move(new_population);
}

std::vector<Individual> GeneticAlgorithm::tournament_selection(size_t count, size_t tournament_size) {
    std::vector<Individual> selected;
    selected.reserve(count);
    
    std::uniform_int_distribution<size_t> idx_dist(0, population_.size() - 1);
    
    for (size_t i = 0; i < count; ++i) {
        Individual best;
        best.fitness = -1.0f; // Ensure any real fitness is better
        
        for (size_t j = 0; j < tournament_size; ++j) {
            size_t idx = idx_dist(rng_);
            if (population_[idx].fitness > best.fitness) {
                best = population_[idx];
            }
        }
        
        selected.push_back(best);
    }
    
    return selected;
}

void GeneticAlgorithm::save_generation_results(uint32_t generation) {
    std::stringstream filename;
    filename << params_.output_directory << "/generation_" << generation << ".csv";
    
    std::ofstream file(filename.str());
    if (!file.is_open()) {
        std::cerr << "Failed to save generation results to " << filename.str() << std::endl;
        return;
    }
    
    file << "individual,fitness,ticks_survived,distance_moved,fruits_eaten,layout_encoding\n";
    for (size_t i = 0; i < population_.size(); ++i) {
        const auto& ind = population_[i];
        file << i << "," << ind.fitness << "," << ind.ticks_survived << "," 
             << ind.distance_moved << "," << ind.fruits_eaten << "," 
             << ind.layout_encoding << "\n";
    }
    
    file.close();
}

void GeneticAlgorithm::save_best_individual(const Individual& best) {
    std::string filename = params_.output_directory + "/best_individual.txt";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to save best individual to " << filename << std::endl;
        return;
    }
    
    file << "Best Individual Results:\n";
    file << "Fitness: " << best.fitness << "\n";
    file << "Ticks Survived: " << best.ticks_survived << "\n";
    file << "Distance Moved: " << best.distance_moved << "\n";
    file << "Fruits Eaten: " << best.fruits_eaten << "\n";
    file << "Layout Encoding: " << best.layout_encoding << "\n";
    
    file.close();
    
    std::cout << "Best individual saved to " << filename << std::endl;
}

} // namespace genetic_algorithm