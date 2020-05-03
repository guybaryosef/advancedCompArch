/*
 * Calls the CPU (sequential) implementation of the GPU-SFFT function
 * and returns the MSE as well as the time it took to run.
 *
 * Usage:
 *     - ./gpuSfft INPUT_FILE OUTPUT_FILE [BIN_COUNT] [NUM_OF_NONE_ZERO_ELEMENTS]
 */


#include <iostream>
#include <string>
#include <optional>
#include <vector>
#include <chrono>

#include "../utils/AlgoParams.hpp"
#include "Algo.hpp"


int main(int argv, char **argc)
{
    AlgoParams algo_params(argv, argc);
    if ( !algo_params.isValid() )
        return -1;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> output_vec = Algo::outerLoop(
             algo_params.getInputVec(),
             algo_params.getFilter_t(),
             algo_params.getFilter_f(),
             algo_params.getB(),
             2*algo_params.getK(),
             algo_params.getW(),
             algo_params.getL(),
             algo_params.getL_c(),
             algo_params.getL_t(),
             algo_params.getL_l() );
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds." << std::endl;
              
    algo_params.writeOutput(output_vec);



    return 0;
}