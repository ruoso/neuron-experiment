#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <vector>
#include <string>
#include <random>
#include <cstdint>

namespace genetic_algorithm {

struct GAParameters {
    uint32_t population_size;
    uint32_t num_generations;
    float mutation_rate;
    float crossover_rate;
    uint32_t elite_size;
    uint32_t random_seed;
    std::string output_directory;
    
    GAParameters() 
        : population_size(20), num_generations(50), mutation_rate(0.1f), 
          crossover_rate(0.7f), elite_size(2), random_seed(12345),
          output_directory("ga_output") {}
};

struct Individual {
    std::string layout_encoding;
    float fitness;
    uint32_t ticks_survived;
    float distance_moved;
    uint32_t fruits_eaten;
    
    Individual() : fitness(0.0f), ticks_survived(0), distance_moved(0.0f), fruits_eaten(0) {}
    Individual(const std::string& encoding) 
        : layout_encoding(encoding), fitness(0.0f), ticks_survived(0), 
          distance_moved(0.0f), fruits_eaten(0) {}
};

class GeneticAlgorithm {
private:
    GAParameters params_;
    std::vector<Individual> population_;
    std::mt19937 rng_;
    size_t current_generation_;
    
    // Genetic operators
    std::string generate_random_layout_encoding();
    std::string mutate_layout(const std::string& layout, float mutation_rate);
    std::string crossover_layouts(const std::string& parent1, const std::string& parent2);
    
    // Simulation and fitness
    float evaluate_individual(Individual& individual, size_t generation, size_t individual_index);
    bool run_simulation(const std::string& layout_encoding, const std::string& output_file);
    float calculate_fitness(uint32_t ticks_survived, float distance_moved, uint32_t fruits_eaten);
    
    // Population management
    void initialize_population();
    void evaluate_population();
    void select_and_reproduce();
    std::vector<Individual> tournament_selection(size_t count, size_t tournament_size);
    
    // Utilities
    void save_generation_results(uint32_t generation);
    void save_best_individual(const Individual& best);

public:
    GeneticAlgorithm(const GAParameters& params);
    ~GeneticAlgorithm() = default;
    
    void run();
    void resume_from_results();
    bool load_population_from_generation(size_t generation);
    size_t find_latest_generation();
    void load_parameters_from_file(const std::string& filename);
    void save_parameters_to_file(const std::string& filename) const;
};

} // namespace genetic_algorithm

#endif // GENETIC_ALGORITHM_H