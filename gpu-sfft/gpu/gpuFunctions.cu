/**
 * gpuFunctions.cu - The CUDA kernels that the GPU
 * implementation uses.
 */

 
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
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < W; idx += stride)
            dx_prime[idx] = dx[tau + idx*sigma];
}


__global__
void S_kernel(
        const   int            *d_yhat, 
                int            *dsamples_s, 
                int            *dsamples_I, 
        const   unsigned        m)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < m; idx += stride)
    {
        dsamples_s[idx] = d_yhat[idx]*d_yhat[idx];
        dsamples_I[idx] = dsamples_s[idx];
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
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < m; idx += stride)
        if (dsamples_I[idx] > cutoff && *id < B_t)
            dId[atomicAdd(id, 1)] = idx;
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
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx <=B; idx += stride)
    {
        if (idx < B)
            for (int j=0; j<T; ++j)
            {
                int id = idx+B;
                dbins_t[idx] += dx[ (id*dH_sig_i) % n] * dfilt_t[id];            
            }
        else if (idx == B)
            for (int j=0; j<R; ++j)
            {
                int id = j*T + idx;
                dbins_t[j] += dx[ (id+dH_sig_i) % n] * dfilt_t[id];
            }        
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
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < fs; idx += stride)
        dbins_t[idx%B] += dx[ (idx*dH_sig_i)%n ] * dfilt_t[idx];
}


__global__
void dot(
        const   int            *a, 
        const   int            *b, 
                int            *c,
        const   int             N)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < N; idx += stride)
        atomicAdd(c, a[idx]*b[idx]);
}

__global__
void makedJ2sig_kernel(
                int            *dJ_2sig, 
        const   int            *dJ2, 
        const   int             B_t,
        const   int             sigma, 
        const   int             W)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < B_t; idx += stride)
        dJ_2sig[idx] = (dJ2[idx]*sigma) % W;
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
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < B_t; idx += stride)
    {
        const int i_Li = dJ[idx];
        atomicAdd(&dV[i_Li], 1);
        if (dV[i_Li] == L_t)
            dI[atomicAdd(IF,1)] = i_Li;
    }
}


__global__
void EV_kernel(
                int            *dx_hat, 
        const   int            *dI, 
        const   int             hIF, 
        const   int            *dbins_f, 
        const   unsigned        L, 
        const   unsigned        n, 
        const   int            *dH_sig, 
        const   unsigned        B, 
        const   int            *dfilt_f)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < hIF; idx += stride)
    {        
        int sum = 0;
        for (int j=3; j < L; ++j)
        {
            int id      = (dH_sig[j]*dI[idx]) % n;
            sum += dbins_f[(dI[idx])/(n/B)] / dfilt_f[id % (n/B)];
        }
        dx_hat[idx] = sum/L;
    }
}


__global__
void intToCureal_kernel(
                cufftReal  *dft_input, 
        const   int        *dinvec_t, 
        const   int         B)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < B; idx += stride)
        dft_input[idx] = dinvec_t[idx];
}


__global__
void curealToInt_kernel(
        const   cufftComplex   *dft_output, 
                int            *dinvec_f, 
        const   int             N)
{
    int start_idx         = blockDim.x*blockIdx.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;

    for (int idx = start_idx; idx < N/2+1; idx += stride)
    {
        if (N/2+idx < N)
            dinvec_f[N/2+idx] = cuCabsf(dft_output[idx]);
        if (N/2-idx >= 0)
			dinvec_f[N/2-idx] = cuCabsf(dft_output[idx]);
	}
}
