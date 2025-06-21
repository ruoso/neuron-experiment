#include "genetic_algorithm.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace genetic_algorithm;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --config <file>     Load GA parameters from config file\n";
    std::cout << "  --population <N>    Population size (default: 20)\n";
    std::cout << "  --generations <N>   Number of generations (default: 50)\n";
    std::cout << "  --mutation <rate>   Mutation rate 0.0-1.0 (default: 0.1)\n";
    std::cout << "  --crossover <rate>  Crossover rate 0.0-1.0 (default: 0.7)\n";
    std::cout << "  --elite <N>         Elite size (default: 2)\n";
    std::cout << "  --seed <N>          Random seed (default: 12345)\n";
    std::cout << "  --output <dir>      Output directory (default: ga_output)\n";
    std::cout << "  --help              Show this help message\n";
}

GAParameters parse_command_line(int argc, char* argv[]) {
    GAParameters params;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--config" && i + 1 < argc) {
            // Config file parsing would go here
            std::cout << "Config file support not yet implemented\n";
            ++i;
        } else if (arg == "--population" && i + 1 < argc) {
            params.population_size = std::stoul(argv[++i]);
        } else if (arg == "--generations" && i + 1 < argc) {
            params.num_generations = std::stoul(argv[++i]);
        } else if (arg == "--mutation" && i + 1 < argc) {
            params.mutation_rate = std::stof(argv[++i]);
        } else if (arg == "--crossover" && i + 1 < argc) {
            params.crossover_rate = std::stof(argv[++i]);
        } else if (arg == "--elite" && i + 1 < argc) {
            params.elite_size = std::stoul(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            params.random_seed = std::stoul(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            params.output_directory = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return params;
}

void validate_parameters(const GAParameters& params) {
    if (params.population_size == 0) {
        throw std::invalid_argument("Population size must be greater than 0");
    }
    if (params.num_generations == 0) {
        throw std::invalid_argument("Number of generations must be greater than 0");
    }
    if (params.mutation_rate < 0.0f || params.mutation_rate > 1.0f) {
        throw std::invalid_argument("Mutation rate must be between 0.0 and 1.0");
    }
    if (params.crossover_rate < 0.0f || params.crossover_rate > 1.0f) {
        throw std::invalid_argument("Crossover rate must be between 0.0 and 1.0");
    }
    if (params.elite_size >= params.population_size) {
        throw std::invalid_argument("Elite size must be less than population size");
    }
}

int main(int argc, char* argv[]) {
    try {
        GAParameters params = parse_command_line(argc, argv);
        validate_parameters(params);
        
        std::cout << "Creature Evolution - Genetic Algorithm" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        GeneticAlgorithm ga(params);
        ga.run();
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}