/**
 * cpuFunctions.h - Function declarations of the 
 * host-side functions that the GPU implementation 
 * uses.
 */


#pragma once

#include <vector>
#include <cuda_runtime.h>


///////// FUNCTION DEFINITIONS /////////
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

void gpu_fftCutoff(
			int				   *dvec,      	// input vector in freq domain after cutoff
	const   int			   	   *dinvec_t,  	// input vector in the time domain
			int				   *dinvec_f,  	// input vector in the freq domain
	const   int                 n_bins,   	// number of bins (length of input vector in both time & freq)
	const   unsigned            B_t);		// 2k (length of dvec)


std::vector<int> gpu_outerLoop(
    const   std::vector<int>    hx,         // input signal
    const   std::vector<int>    dfilter_t,  // time component of filter
    const   std::vector<int>    dfilter_f,  // freq component of filter
    const   unsigned            fs,         // length of filter
    const   unsigned            B,          // number of bins
    const   unsigned            B_t,        // 2k (k== number of non-zero freq coeff in output vector)
    const   unsigned            W,          // the length of some intermediate vectors
    const   unsigned            L,          // number of loops in outside loop
    const   unsigned            L_c,        // number of loops (times to execute) the locLargeCoef function
    const   unsigned            L_t,        // loop threshold for the revHash function
	const   unsigned            L_l);       // threshold of whether or not to implement the revHash function
	

void gpu_locLargeCoefGPU(
	const   int                *dx,         // input signal
	const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
	const   unsigned            n,          // the length of the input vector
	const   unsigned            W,          // the length of the output vector
			int                *dJ2);      	// function output


void gpu_cutOff(
			int                *dId,        // the B_t largest freq coeffs in the bins vector in freq domain
	const   unsigned            B_t,        // 2k (length of dId)
			int                *d_yhat,     // vector in freq domain
	const   unsigned            m);      	// the length of d_yhat

void gpu_permFilter(
	const   int   			   *dx,         // input signal
	const	int					n,			// length of the input signal
	const   int				   *dfilt_t,    // time component of filter
	const	int					fs,			// the length of the filter
			int				   *dbins_t,    // permuted and filtered components of input signal are binned here
	const   unsigned            B,          // number of bins
	const   int                 dH_sig_i); 	// modular inverse of sigma and n
	
void gpu_revHash(
			int			   *dI, 
	const	int			   *dJ, 
	const	unsigned		B_t, 
	const	unsigned		B, 
	const	unsigned		n, 
	const	unsigned		L_t, 
	const	int			   *dJ2, 
	const	unsigned		W, 
			int			   *IF, 
	const	int				sigma);

int *gpu_eval(
	const	int			   *dI, 
	const	int			    hIF, 
	const	int			   *dbins_f, 
	const	int			   *dfilter_f, 
	const	unsigned		B, 
	const	unsigned		n, 
	const	unsigned		L, 
	const	int			   *dH_sig);