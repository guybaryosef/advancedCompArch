/**
 * Algo.cpp - Header File for the
 * main algorithm class.
 *
 * The function names and parameters are derived
 * directly from the GPU-SFFT paper. Even though not
 * all the parameters are actually needed in the CPU version,
 * they were included because the goal of this project
 * is to have as close to an identical implementation
 * for both the CPU and GPU versions.
 */

#pragma once

#include <vector>



class Algo
{
private:
public:
    Algo() {};

    static std::vector<int> outerLoop(
            const   std::vector<int>    hx,         // input signal
            const   std::vector<int>    dfilter_t,  // time component of filter
            const   std::vector<int>    dfilter_f,  // freq component of filter
            const   unsigned            B,          // number of bins
            const   unsigned            B_t,        // 2k (k== number of non-zero freq coeff in output vector)
            const   unsigned            W,          // the length of some intermediate vectors
            const   unsigned            L,          // number of loops in outside loop
            const   unsigned            L_c,        // number of loops (times to execute) the locLargeCoef function
            const   unsigned            L_t,        // loop threshold for the revHash function
            const   unsigned            L_l);       // threshold of whether or not to implement the revHash function

    static void permFilter(
            const   std::vector<int>&   x,          // input signal
            const   unsigned            B,          // number of bins
            const   std::vector<int>&   filt_t,     // time component of filter
                    std::vector<int>&   bin_st,     // permuted and filtered components of input signal are binned here
            const   int                 H_sig_i);   // modular inverse of sigma and n

    static void fftCutoff(
                    std::vector<int>&   dJ,         // the B_t largest freq coeffs in the bins vector in freq domain
            const   std::vector<int>&   dbins_t,    // the bins vector in the time domain
                    std::vector<int>&   dbins_f,    // the bins vector in the freq domain
            const   unsigned            B_t);       // 2k (k== number of non-zero freq coeffs in output vector)

    static void cutOff(
                    std::vector<int>&   dId,        // the B_t largest freq coeffs in the bins vector in freq domain
            const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
                    std::vector<int>&   d_yhat);    // the bins vector in the freq domain

    static void locLargeCoef(
            const   std::vector<int>&   dx,         // input signal
            const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
            const   unsigned            W,          // the length of the output vector
                    std::vector<int>&   dJ2,        // function output
            const   int                 i);         // the index that dJ2 is up to

    static void revHash(
                    std::vector<int>&   dI,         // output signal
            const   std::vector<int>&   dJ,         // input 1
            const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
            const   unsigned            B,          // number of bins
            const   unsigned            n,          // length of sfft input signal
            const   unsigned            L_t,        // loop threshold for the revHash function
            const   std::vector<int>&   dJ_2,       // input 2
            const   unsigned            W,          // length of dj_2
                    unsigned&           IF,         // output signal length
            const   unsigned            sigma);     // constant


    static std::vector<int> eval(
        const   std::vector<int>&   dI,         // an input vector
        const   unsigned            IF,         // length of output vector
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   std::vector<int>&   dfilter_f,  // freq component of filter
        const   unsigned            B,          // number of bins
        const   unsigned            n,          // length of sfft input signal
        const   unsigned            L,          // number of loops in outside loop
        const   std::vector<int>&   dH_sig);    // some vector
};