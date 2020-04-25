/**
 * Algo.cpp - Implementation File for the
 * main algorithm class.
 */

#include "Algo.hpp"



std::vector<int> Algo::outerLoop(
        const   std::vector<int>    hx,         // input signal
        const   unsigned            n,          // size of input signal
        const   std::vector<int>    dfilter_t,  // time component of filter
        const   std::vector<int>    dfilter_f,  // freq component of filter
        const   unsigned            fs,         // length of filter
        const   unsigned            B,          // number of bins
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeff in output vector)
        const   unsigned            W,          // the length of some intermediate vectors
        const   unsigned            L,          // number of loops in outside loop
        const   unsigned            L_c,        // number of loops (times to execute) the locLargeCoef function
        const   unsigned            L_t,        // loop threshold for the revHash function
        const   unsigned            L_l)        // threshold of whether or not to implement the revHash function
{

}


void Algo::permFilter(
        const   std::vector<int>&   x,          // input signal
        const   unsigned            B,          // number of bins
        const   std::vector<int>&   filt_t,     // time component of filter
        const   unsigned            fs,         // filter length
                std::vector<int>&   bin_st,     // permuted and filtered components of input signal are binned here
        const   int                 H_sig_i)    // modular inverse of sigma and n
{}


void Algo::fftCutoff(
                std::vector<int>&   dJ,         // the B_t largest freq coeffs in the bins vector in freq domain
        const   std::vector<int>&   dbins_t,    // the bins vector in the time domain
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   unsigned            B,          // number of bins
        const   unsigned            B_t)        // 2k (k== number of non-zero freq coeffs in output vector)
{


}


void Algo::cutOff(
                std::vector<int>&   dId,        // the B_t largest freq coeffs in the bins vector in freq domain
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
                std::vector<int>&   d_yhat,     // the bins vector in the freq domain
        const   unsigned            m)          // the length of d_yhat
{

}


void Algo::S(
                std::vector<int>&   d_yhat,     // the bins vector in the freq domain
                std::vector<int>&   dsamples_ss,// first  output of function
                std::vector<int>&   dsamples_sI,// second output of function
        const   unsigned            m)          // the length of d_yhat
{

}


void Algo::C(
                std::vector<int>&   dId,        // the B_t largest freq coeffs in the bins vector in freq domain
        const   unsigned            cutoff,     // cutoff value
        const   std::vector<int>&   dsamples_sI,// input vector getting cut-offed
        const   unsigned            m,          // length of dsamples_sI
        const   int                 id,         // constant 0?
        const   unsigned            B_t)        // 2k (k== number of non-zero freq coeffs in output vector)
{

}


void Algo::locLargeCoef(
        const   std::vector<int>&   dx,         // input signal
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
        const   unsigned            n,          // the length of the input vector
        const   unsigned            W,          // the length of the output vector
                std::vector<int>&   dJ2)        // function output
{

}


void Algo::LLC(
                std::vector<int>&   dx_o,       // function output
        const   std::vector<int>&   dx,         // input signal
        const   unsigned            W,          // output length
        const   unsigned            tau,        // constant
        const   unsigned            sigma)      // constant
{

}


void Algo::revHash(
                std::vector<int>&   dI,         // output signal
        const   std::vector<int>&   dJ,         // input 1
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
        const   unsigned            B,          // number of bins
        const   unsigned            n,          // the length of dJ
        const   unsigned            L_t,        // loop threshold for the revHash function
        const   std::vector<int>&   dJ_2,       // input 2
        const   unsigned            W,          // length of dj_2
        const   unsigned            IF,         // output signal length
        const   unsigned            sigma)      // constant
{

}


void Algo::RH(
                std::vector<int>&   dI,         // output signal
        const   std::vector<int>&   dJ,         // input vector 1
                std::vector<int>&   dV,         // a vector that starts off as zeroed out
        const   std::vector<int>&   dJ_2sig,    // input vector 2
        const   unsigned            L_t,        // loop threshold for the revHash function
        const   unsigned            IF,         // output signal length
        const   unsigned            W)          // length of dj_2
{

}


std::vector<int> Algo::eval(
        const   std::vector<int>&   dI,         // an input vector
        const   unsigned            IF,         // length of output vector
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   std::vector<int>    dfilter_f,  // freq component of filter
        const   unsigned            B,          // number of bins
        const   unsigned            n,          // length of filter
        const   unsigned            L,          // number of loops in outside loop
        const   std::vector<int>&   dH_sig)     // some vector
{

}


void Algo::EV(
                std::vector<int>&   dx_hat,     // output vector
        const   std::vector<int>&   dI,         // an input vector
        const   unsigned            IF,         // length of output vector
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   unsigned            L,          // number of loops in outside loop
        const   unsigned            n,          // length of filter
        const   std::vector<int>&   dH_sig,     // some vector
        const   unsigned            B,          // number of bins
        const   std::vector<int>    dfilter_f)  // freq component of filter
{

}