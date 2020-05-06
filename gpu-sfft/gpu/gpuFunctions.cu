

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cufft.h>

#include "../utils/AlgoParams.hpp"


__global__
void LLC_kernel(
                int                *dx_prime,   // function output
        const   int                *dx,         // input signal
        const   unsigned            W,          // output length
        const   unsigned            tau,        // constant
        const   unsigned            sigma)      // constant
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < W)
        dx_prime[i] = dx[tau + i*sigma];
}


__global__
void S_kernel(
        const   int            *d_yhat, 
                int            *dsamples_s, 
                int            *dsamples_I, 
        const   unsigned        m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < m)
    {
        dsamples_s[i] = d_yhat[i]*d_yhat[i];
        dsamples_I[i] = dsamples_s[i];
    }
}


__global__
void C_kernel(
                int            *dId, 
        const   int             cutoff, 
        const   int            *dsamples_I, 
        const   int             m, 
                int            *id, 
        const   unsigned        B_t)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < m)
    {
        if (dsamples_I[i] > cutoff && *id < B_t)
        {
            dId[atomicAdd(id, 1)] = i;
        }
    }
}


__global__
void PFT_kernel(
                int            *dbins_t, 
        const   int            *dx, 
        const   int            *dfilt_t, 
        const   unsigned        n, 
        const   unsigned        B, 
        const   int             dH_sig_i, 
        const   int             T, 
        const   int             R)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i<B)
        for (int j=0; j<T; ++j)
        {
            int id = i+B;
            dbins_t[i] += dx[ (id*dH_sig_i) % n] * dfilt_t[id];            
        }
    else if (i==B)
        for (int j=0; j<R; ++j)
        {
            int id = j*T + i;
            dbins_t[j] += dx[ (id+dH_sig_i) % n] * dfilt_t[id];
        }        
}


__global__
void PFK_kernel(
                int            *dbins_t, 
        const   int            *dx, 
        const   int            *dfilt_t, 
        const   unsigned        n, 
        const   unsigned        B, 
        const   int             dH_sig_i, 
        const   unsigned        fs)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < fs)
        dbins_t[i%B] += dx[ (i*dH_sig_i)%n ] * dfilt_t[i];
}


__global__
void dot(
        const   int            *a, 
        const   int            *b, 
                int            *c,
        const   int             N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < N)
        atomicAdd(c, a[i]*b[i]);
}


__global__
void RH_kernel(
                int            *dI, 
        const   int            *dJ, 
                int            *dV, 
        const   int            *dJ_2sig, 
        const   unsigned        L_t, 
                int            *IF, 
        const   unsigned        B_t)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < B_t)
    {
        const int i_Li = dJ[i];
        atomicAdd(&dV[i_Li], 1);
        if (dV[i_Li] == L_t)
            dI[atomicAdd(IF,1)] = i_Li;
    }
}


__global__
void EV_kernel(
                int            *dx_hat, 
        const   int            *dI, 
        const   int            *IF, 
        const   int            *dbins_f, 
        const   unsigned        L, 
        const   unsigned        n, 
        const   int            *dH_sig, 
        const   unsigned        B, 
        const   int            *dfilt_f)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < IF[0])
    {
        int x_hat_v[DEFAULT_L];
        for (int j=0, pos=0; j < L; ++j)
        {
            int id = (dH_sig[j]*dI[i]) % n;
            x_hat_v[pos] = dbins_f[(j/L)*B] / dfilt_f[id % (n/B)];
            ++pos;
        }

        thrust::sort(thrust::seq, x_hat_v, x_hat_v + L);
        dx_hat[i] = x_hat_v[L/2];
    }
}


__global__
void intToCureal_kernel(
                cufftReal  *dft_input, 
        const   int        *dinvec_t, 
        const   int         n_bins)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n_bins)
        dft_input[i] = dinvec_t[i];
}


__global__
void curealToInt_kernel(
        const   cufftComplex   *dft_output, 
                int            *dinvec_f, 
        const   int             N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < N/2+1)
    {
        if (N/2+i < N)
            dinvec_f[N/2+i] = cuCabsf(dft_output[i]);
        if (N/2-i >= 0)
			dinvec_f[N/2-i] = cuCabsf(dft_output[i]);
	}

}
