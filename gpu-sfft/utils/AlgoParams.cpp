/**
 * AlgoParams.cpp - Implementation File for the
 * algorithm parameters class.
 */

#include "AlgoParams.hpp"



AlgoParams::AlgoParams(const int argv, char **argc)
{
    bool err = false;
    n_bins = DEFAULT_N_BIN;
    k      = DEFAULT_K;

    if (argv == 1 || argv > 5)
        err = true;
    if (argv >= 2)
    {
        try        { input_name = argc[1];  }
        catch(...) { err = true;            }
    }
    if (argv >= 3)
    {
        try        { output_name = argc[2]; }
        catch(...) { err = true;            }
    }
    if (argv >= 4)
    {
        try         {  n_bins = std::stoi(argc[3]); }
        catch(...)  {  err = true;                  }
    }
    if (argv == 5)
    {
        try         {  k   = std::stoi(argc[4]); }
        catch(...)  {  err = true;               }
    }

    if (err)
    {
        std::cerr << "Correct Usage: ./gpuSfft INPUT_FILE OUTPUT_FILE [BIN_COUNT] [NUM_OF_NONE_ZERO_ELEMENTS]" << std::endl;
        valid = false;
    }
    else
        valid = true;

    loadInputVectorFromFile(input_name);
};


void AlgoParams::loadInputVectorFromFile(const std::string &file_name)
{
    input_vec.empty();  /* zero out (a possibly old) input vector */

    std::ifstream f_in(file_name);

    int cur_val;
    while (f_in >> cur_val)
        input_vec.push_back(cur_val);
}



