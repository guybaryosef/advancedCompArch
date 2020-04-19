/*
 * Calls the GPU-SFFT function.
 *
 * Usage:
 *     - ./gpuSfft INPUT_FILE OUTPUT_FILE [BIN_COUNT] [NUM_OF_NONE_ZERO_ELEMENTS]
 */


#include <iostream>
#include <string>
#include <optional>
#include <vector>

#include "../utils/AlgoParams.hpp"
#include "Algo.hpp"


int main(int argv, char **argc)
{
    AlgoParams algo_params(argv, argc);
    if ( !algo_params.isValid() )
        return -1;


//    std::cout << "bins: " << algo_params.getNBins() << " | K: " << algo_params.getK() << std::endl;
//    std::vector<int> input = algo_params.getInputVec();
//    std::cout << "Input: ";
//    for (const auto &ele : input)
//        std::cout << ele << " ";
//    std::cout << std::endl;



    return 0;
}





std::vector<int> parseInput(std::string &file_name)
{

}