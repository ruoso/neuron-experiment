# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Build: `cmake --build build/`
- Run tests: `make test` or `ctest`
- Run single test: `ctest -R <test_name>` or `./bin/test_<test_name>`
- Lint: `make lint` or `cpplint --filter=-legal/copyright src/**/*.cpp include/**/*.h`

## Code Style Guidelines
- **Formatting**: 4-space indentation, 100 character line limit
- **Naming**: 
  - Classes/Structs: PascalCase (e.g., `DendriticBranch`)
  - Variables/Functions: snake_case (e.g., `output_targets`)
  - Constants: UPPER_SNAKE_CASE
- **Types**: Use explicit types. Prefer strongly typed enums, `uint32_t` for addresses
- **Imports**: Group system headers first, then project headers
- **Error Handling**: Use exceptions for exceptional conditions, return codes for expected errors
- **Memory Management**: Prefer RAII and smart pointers over manual memory management
- **Documentation**: Document all public functions and non-obvious implementations
- **Testing**: Write unit tests for all core functionality
- Always use include guards instead of `#pragma once`
- Always create .cpp/.h pairs even if the cpp file is empty

## Design Principles
- We will follow a functional pattern, so data types will have just data, and then we will implement pure functions
