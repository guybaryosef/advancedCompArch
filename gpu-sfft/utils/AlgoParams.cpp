/**
 * AlgoParams.cpp - Implementation File for the
 * algorithm parameters class.
 */


#include <cmath>
#include <fftw3.h>  // fftw (fft C subroutine library)
#include <string>
#include <sstream>

#include "AlgoParams.hpp"


AlgoParams::AlgoParams(int argv, char **argc)
    :
    B(DEFAULT_B),
    K(DEFAULT_K),
    W(DEFAULT_W),
    L(DEFAULT_L),
    L_c(DEFAULT_L_c),
    L_t(DEFAULT_L_t),
    L_l(DEFAULT_L_l)
{
    bool err = false;

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
        try         {  B = std::stoi(argc[3]);  }
        catch(...)  {  err = true;              }
    }
    if (argv == 5)
    {
        try         {  K   = std::stoi(argc[4]); }
        catch(...)  {  err = true;               }
    }

    if (err)
    {
        std::cerr << "Correct Usage: ./gpuSfft INPUT_FILE OUTPUT_FILE [BIN_COUNT] [NUM_OF_NONE_ZERO_ELEMENTS]" << std::endl;
        valid = false;
    }
    else
        valid = true;

    loadInputVectorFromFile();
    createFilter();
};


void AlgoParams::loadInputVectorFromFile()
{
    input_vec.empty();  /* zero out (a possibly old) input vector */

    std::ifstream   f_in(input_name);
    std::string     line;

    while ( std::getline(f_in, line) )
    {
        std::stringstream ss(line);
        
        int cur_val;
        while( ss >> cur_val)
            input_vec.push_back(cur_val);
    }

    f_in.close();
}


void AlgoParams::writeOutput(const std::vector<int> &ouput_vec)
{
    std::ofstream f_out(output_name);

    for (const auto &ele : output_vec)
        f_out << ele << " ";
    f_out << std::endl;

    f_out.close();
}


void AlgoParams::createFilter()
{
    // create the gaussian filter
    std::vector<double> gaussianFilter(FILTER_SIZE, 0);
    double sum = 0.0;
    for (int i = -FILTER_SIZE/2; i <= FILTER_SIZE/2; ++i)
    {
        gaussianFilter[i + FILTER_SIZE/2] = exp(-(i*i)) / 2;
        sum += gaussianFilter[i + FILTER_SIZE/2];
    }
    for (int i = 0; i < FILTER_SIZE; ++i)
        gaussianFilter[i] = 255*(gaussianFilter[i]/255);

    // create the box filter
    std::vector<int> rectangularFilter(FILTER_SIZE, 0);
    for (int i = FILTER_SIZE/4; i<3*FILTER_SIZE/4; ++i)
        rectangularFilter[i] = 1;

    // create the filter
    filter_f = std::vector<int>(FILTER_SIZE, 0);

    for (int n=0; n<FILTER_SIZE; ++n)
        for (int m = n; m < FILTER_SIZE; ++m)
            filter_f[n] = static_cast<int>( gaussianFilter[m] * rectangularFilter[m-n] );

    // now take the ifft to get the time domain representation of the signal.
    double *in  = (double*) fftw_malloc(sizeof(double)*FILTER_SIZE);
    for (int i=0; i<FILTER_SIZE; ++i)
        in[i] = filter_f[i];

    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (FILTER_SIZE/2 + 1));

    fftw_plan p = fftw_plan_dft_r2c_1d(FILTER_SIZE, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    filter_t = std::vector<int>(FILTER_SIZE, 0);
    for (int i=0; i<(FILTER_SIZE/2+1); ++i)
    {
        if (FILTER_SIZE/2+i < FILTER_SIZE)
            filter_t[FILTER_SIZE/2+i] = out[i][0];
        if (FILTER_SIZE/2-i >= 0)
            filter_t[FILTER_SIZE/2-i] = out[i][0];
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}