/**
 * Algo.cpp - Implementation File for the
 * main algorithm class.
 */


#include <stdlib.h>
#include <algorithm>

#include <fftw3.h>  // fftw (fft C subroutine library)

#include "Algo.hpp"
#include "../utils/otherFunctions.cpp"


std::vector<int> Algo::outerLoop(
        const   std::vector<int>    hx,         // input signal
        const   std::vector<int>    dfilter_t,  // time component of filter
        const   std::vector<int>    dfilter_f,  // freq component of filter
        const   unsigned            B,          // number of bins
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeff in output vector)
        const   unsigned            W,          // the length of some intermediate vectors
        const   unsigned            L,          // number of loops in outside loop
        const   unsigned            L_c,        // number of loops (times to execute) the locLargeCoef function
        const   unsigned            L_t,        // loop threshold for the revHash function
        const   unsigned            L_l)        // threshold of whether or not to implement the revHash function
{
    std::vector<int> dbins_t(B,         0);
    std::vector<int> dbins_f(B,         0);
    std::vector<int> dI     (hx.size(), 0);
    std::vector<int> dJ_2   (L_c*B_t,   0);
    std::vector<int> dH_sig (L,         0);

    for (int i=0; i<L_c; ++i)
        locLargeCoef(hx, B_t, W, dJ_2, i);

    unsigned IF = 0;
    for (int i=0; i<L; ++i)
    {
        int sigma = rand() % hx.size();
        dH_sig[i] = modInverse(sigma, hx.size());

        std::vector<int> dJ(B_t, 0);

        permFilter(hx, B, dfilter_t, dbins_t, dH_sig[i]);
        fftCutoff(dJ, dbins_t, dbins_f, B_t);

        if (i < L_l)
                revHash(dI, dJ, B_t, B, hx.size(), L_t, dJ_2, W, IF, sigma);
    }

    std::vector<int> output = eval(dI, IF, dbins_f, dfilter_f, B, hx.size(), L, dH_sig);

    return output;
}


void Algo::permFilter(
        const   std::vector<int>&   x,          // input signal
        const   unsigned            B,          // number of bins
        const   std::vector<int>&   filt_t,     // time component of filter
                std::vector<int>&   bin_st,     // permuted and filtered components of input signal are binned here
        const   int                 H_sig_i)    // modular inverse of sigma and n
{
    for (int i=0, index=0; i<filt_t.size(); ++i)
    {
        bin_st[i%B] += x[index]*filt_t[i];
        index = (index+H_sig_i) % x.size();
    }
}


void Algo::fftCutoff(
                std::vector<int>&   dJ,         // the B_t largest freq coeffs in the bins vector in freq domain
        const   std::vector<int>&   dbins_t,    // the bins vector in the time domain
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   unsigned            B_t)        // 2k (k== number of non-zero freq coeffs in output vector)
{
    int B = dbins_t.size();

    // use the FFTW library for a real-valued 1-D fft
    double *in  = (double*) fftw_malloc(sizeof(double)*B);
    for (int i=0; i<B; ++i)
        in[i] = dbins_t[i];

    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (B/2 + 1));

    fftw_plan p = fftw_plan_dft_r2c_1d(B, in, out, FFTW_ESTIMATE);

    fftw_execute(p);

    for (int i=0; i<(B/2+1); ++i)
    {
        if (B/2+i < B)
            dbins_f[B/2+i] = out[i][0];
        if (B/2-i >= 0)
            dbins_f[B/2-i] = out[i][0];
    }

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    cutOff(dJ, B_t, dbins_f);
}


void Algo::cutOff(
                std::vector<int>&   dId,        // the B_t largest freq coeffs in the bins vector in freq domain
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
                std::vector<int>&   d_yhat)     // the bins vector in the freq domain
{
    std::vector<int> dsamples_s(d_yhat.size(),0);
    std::vector<int> dsamples_I(d_yhat.size(),0);

    for (int i=0; i<d_yhat.size(); ++i)
    {
        dsamples_s[i] = d_yhat[i] * d_yhat[i];
        dsamples_I[i] = dsamples_s[i];
    }

    std::sort(dsamples_s.begin(), dsamples_s.end());

    int cutoff = dsamples_s[d_yhat.size() - B_t - 1];
    for (int i=0, id=0; i<d_yhat.size(); ++i)
        if (dsamples_I[i] > cutoff && id < B_t)
            dId[++id] = i;
}


void Algo::locLargeCoef(
        const   std::vector<int>&   dx,         // input signal
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
        const   unsigned            W,          // the length of the output vector
                std::vector<int>&   dJ2,        // function output
        const   int                 i)          // the index that dJ2 is up to
{
    std::vector<int> dx_prime(W, 0);
    std::vector<int> dy_hat  (W, 0);

    int sigma = dx.size()/W;
    int tau   = rand() % sigma;

    for (int i=0; i<W; ++i)
        if (i < W)
            dx_prime[i] = dx[tau + i*sigma];

    fftCutoff(dJ2, dx_prime, dy_hat, B_t);
}


void Algo::revHash(
                std::vector<int>&   dI,         // output signal
        const   std::vector<int>&   dJ,         // input 1
        const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
        const   unsigned            B,          // number of bins
        const   unsigned            n,          // length of sfft input signal
        const   unsigned            L_t,        // loop threshold for the revHash function
        const   std::vector<int>&   dJ_2,       // input 2
        const   unsigned            W,          // length of dj_2
                unsigned&           IF,         // output signal length
        const   unsigned            sigma)      // constant
{
    std::vector<int> dV     (n, 0);
    std::vector<int> dJ_2sig(B_t, 0);

    for (int i=0; i<B_t; ++i)
        dJ_2sig[i] = (dJ_2[i]*sigma) % W;

    std::vector<int> I_L;
    for (int i=0; i<B_t; ++i)
    {
        int iL_i = findLargestFreqCoeff(I_L, dJ, dJ_2sig, n);

        if (++dV[iL_i] == L_t)
            dI[++IF] = iL_i;
    }
}

int Algo::findLargestFreqCoeff(
                std::vector<int>&   I_L,
        const   std::vector<int>&   dJ,
        const   std::vector<int>&   dJ_2sig,
        const   unsigned            n)
{
    int iL_l = 0;
    return iL_l;
}



std::vector<int> Algo::eval(
        const   std::vector<int>&   dI,         // an input vector
        const   unsigned            IF,         // length of output vector
                std::vector<int>&   dbins_f,    // the bins vector in the freq domain
        const   std::vector<int>&   dfilter_f,  // freq component of filter
        const   unsigned            B,          // number of bins
        const   unsigned            n,          // length of sfft input signal
        const   unsigned            L,          // number of loops in outside loop
        const   std::vector<int>&   dH_sig)     // some vector
{
    std::vector<int> dx_hat(IF, 0);

    for (int i=0; i<IF; ++i)
    {
        std::vector<int> x_prime_v(L, 0);

        for (int j=0; j<L; ++j)
        {
            int id = (dH_sig[j]*dI[i]) % n;
            auto max_ele = std::max_element(dbins_f.begin(), dbins_f.end() );
            x_prime_v[j] = *max_ele / dfilter_f[id % (n/B)];
        }

        std::sort(x_prime_v.begin(), x_prime_v.end());
        dx_hat[i] = x_prime_v[ x_prime_v.size()/2];
    }

    return dx_hat;
}