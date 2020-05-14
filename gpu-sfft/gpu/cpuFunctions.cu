/**
 * cpuFunctions.cu - Function definitions of the 
 * host-side functions that the GPU implementation 
 * uses.
 */


#include <math.h>
#include <vector>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

#include "../utils/otherFunctions.cpp"

#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID  2*65535


#include "cpuFunctions.h"


// Error checking macro, copied from:
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_WARN(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


///////// FUNCTION DECLERATIONS /////////
std::vector<int> gpu_outerLoop(
    const   std::vector<int>    hx,         // input signal
    const   std::vector<int>    hfilter_t,  // time component of filter
    const   std::vector<int>    hfilter_f,  // freq component of filter
    const   unsigned            fs,         // length of filter
    const   unsigned            B,          // number of bins
    const   unsigned            B_t,        // 2k (k== number of non-zero freq coeff in output vector)
    const   unsigned            W,          // the length of some intermediate vectors
    const   unsigned            L,          // number of loops in outside loop
    const   unsigned            L_c,        // number of loops (times to execute) the locLargeCoef function
    const   unsigned            L_t,        // loop threshold for the revHash function
    const   unsigned            L_l)        // threshold of whether or not to implement the revHash function
{
	// allocating and copying memory to GPU
	int input_size = hx.size()*sizeof(int);


	int *hx_ptr = new int[input_size];
	
	for (int i=0; i<hx.size(); ++i)
		hx_ptr[i] = hx[i];

	int *hfilter_t_ptr = new int[fs];
	int *hfilter_f_ptr = new int[fs];
	for (int i=0; i<fs; ++i)
	{
		hfilter_t_ptr[i] = hfilter_t[i];
		hfilter_f_ptr[i] = hfilter_f[i];
	}

	int *dx = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dx, input_size) );

    CUDA_WARN( cudaMemcpy(dx, hx_ptr, input_size, cudaMemcpyHostToDevice) );
  
	int *dfilter_t = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dfilter_t, fs*sizeof(int)) ) ;

	CUDA_WARN( cudaMemcpy(dfilter_t, hfilter_t_ptr, fs*sizeof(int), cudaMemcpyHostToDevice) );
	
	int *dfilter_f = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dfilter_f, fs*sizeof(int)) );
	
	CUDA_WARN( cudaMemcpy(dfilter_f, hfilter_f_ptr, fs*sizeof(int), cudaMemcpyHostToDevice) );

	int *dbins_t = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dbins_t, B*sizeof(int)) );

	int *dbins_f = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dbins_f, B*sizeof(int)) );
    
	int *dI = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dI, input_size) );

	int *dJ2 = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dJ2, B_t*sizeof(int)) );

	int *dH_sig = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dH_sig, L*sizeof(int)) );
	int H_sig[L];

	int *IF;
	CUDA_WARN( cudaMalloc(&IF, sizeof(int)) );
	int hIF = 0;
	CUDA_WARN( cudaMemcpy(IF, &hIF, sizeof(int), cudaMemcpyHostToDevice) );
	
	// executing the algorithm
    for (int i=0; i<L_c; ++i)
        gpu_locLargeCoefGPU(dx, B_t, hx.size(), W, dJ2);
	
    for (int i=0; i<L; ++i)
    {
		int sigma = rand() % hx.size();
        H_sig[i] = modInverse(sigma, hx.size());

        int *dJ = nullptr;
		CUDA_WARN( cudaMalloc((void **)&dJ, B_t*sizeof(int)) );

        gpu_permFilter(dx, hx.size(), dfilter_t, fs, dbins_t, B, H_sig[i]);
        gpu_fftCutoff(dJ, dbins_t, dbins_f, B, B_t);

        if (i < L_l)
				gpu_revHash(dI, dJ, B_t, B, hx.size(), L_t, dJ2, W, IF, sigma);
				
		CUDA_WARN( cudaFree(dJ) );
    }

	CUDA_WARN( cudaMemcpy(&hIF, IF, sizeof(int), cudaMemcpyDeviceToHost) );
	
	CUDA_WARN( cudaMemcpy(dH_sig, H_sig, L*sizeof(int), cudaMemcpyHostToDevice) );

	int *output = gpu_eval(dI, hIF, dbins_f, dfilter_f, B, hx.size(), L, dH_sig);

	// freeing all the GPU memory
	CUDA_WARN( cudaFree(dx) );
    
	CUDA_WARN( cudaFree(dbins_t) );
	
	CUDA_WARN( cudaFree(dbins_f) );
	
	CUDA_WARN( cudaFree(dI) );

	CUDA_WARN( cudaFree(dJ2) );
	
	CUDA_WARN( cudaFree(dH_sig) );

	CUDA_WARN( cudaFree(IF) );
	
	CUDA_WARN( cudaFree(dfilter_f) );

	CUDA_WARN( cudaFree(dfilter_t) );

    return std::vector<int>(output, output + hIF);
}


void gpu_locLargeCoefGPU(
	const   int                *dx,         // input signal
	const   unsigned            B_t,        // 2k (k== number of non-zero freq coeffs in output vector)
	const   unsigned            n,          // the length of the input vector
	const   unsigned            W,          // the length of the output vector
			int                *dJ2)        // function output
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int *dx_prime;
	CUDA_WARN( cudaMalloc((void **)&dx_prime, W*sizeof(int)) );
	
	int *dy_hat = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dy_hat, W*sizeof(int)) );

	int sigma = n/W;
	int tau   = rand() % sigma;
	LLC_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx_prime, dx, W, tau, sigma);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch LLC_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	gpu_fftCutoff(dJ2, dx_prime, dy_hat, W, B_t);

	CUDA_WARN( cudaFree(dy_hat) );

	CUDA_WARN( cudaFree(dx_prime) );
}


void gpu_fftCutoff(
			int				   *dvec,      	// input vector in freq domain after cutoff
	const   int			   	   *dinvec_t,  	// input vector in the time domain
			int				   *dinvec_f,  	// input vector in the freq domain
	const   int       			n_bins,   	// number of bins (length of input vector in both time & freq)
	const   unsigned            B_t)        // 2k (length of dvec)
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	// executing cuFFT
	int dft_length= n_bins/2 + 1; 
	int dft_batch = 1;	// number of dfts

	cufftComplex *dft_output;
	CUDA_WARN( cudaMalloc((void**)&dft_output, sizeof(cufftComplex)*dft_length) );

	cufftReal *dft_input;
	CUDA_WARN( cudaMalloc((void**)&dft_input, sizeof(cufftReal)*n_bins) );

	intToCureal_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dft_input, dinvec_t, n_bins);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch intToCureal_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	cufftHandle plan;
	if (cufftPlan1d(&plan, n_bins, CUFFT_R2C, dft_batch) != CUFFT_SUCCESS)
	{
		std::cerr << "CUFFT error: Plan creation failed" << std::endl;
		return;	
	}	

	if (cufftExecR2C(plan, dft_input, dft_output) != CUFFT_SUCCESS)
	{
		std::cerr << "CUFFT error: ExecC2C Forward failed." << std::endl;
		return;	
	}

	CUDA_WARN( cudaDeviceSynchronize() );

	curealToInt_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dft_output, dinvec_f, n_bins);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch curealToInt (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	CUDA_WARN( cudaFree(dft_output) );

	CUDA_WARN( cudaFree(dft_input) );

	cufftDestroy(plan);

	gpu_cutOff(dvec, B_t, dinvec_f, n_bins);
}


void gpu_cutOff(
			int                *dId,        // the B_t largest freq coeffs in the bins vector in freq domain
	const   unsigned            B_t,        // 2k (length of dId)
			int                *d_yhat,     // vector in freq domain
	const   unsigned            m)          // the length of d_yhat
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int *dsamples_s = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dsamples_s, m*sizeof(int)) );
	
	int *dsamples_I = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dsamples_I, m*sizeof(int)) );
	
	S_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_yhat, dsamples_s, dsamples_I, m);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch S_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	thrust::device_ptr<int> dsamples_s_thrust = thrust::device_pointer_cast(dsamples_s);
	thrust::sort(dsamples_s_thrust, dsamples_s_thrust+m);
	dsamples_s = thrust::raw_pointer_cast(dsamples_s_thrust);

	int cutoff;	
	int ind = m-B_t-1;
	CUDA_WARN( cudaMemcpy(&cutoff, dsamples_s + ind, sizeof(int), cudaMemcpyDeviceToHost) );

	int *id = nullptr;
	CUDA_WARN( cudaMalloc((void **)&id, sizeof(int)) );
	
	C_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dId, cutoff, dsamples_I, m, id, B_t);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch C_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	thrust::device_ptr<int> dId_thrust = thrust::device_pointer_cast(dId);
	thrust::sort(dId_thrust, dId_thrust+B_t);
	dId = thrust::raw_pointer_cast(dId_thrust);

	CUDA_WARN( cudaFree(dsamples_s) );

	CUDA_WARN( cudaFree(dsamples_I) );
}


void gpu_permFilter(
	const   int   			   *dx,         // input signal
	const	int					n,			// length of the input signal
	const   int				   *dfilt_t,    // time component of filter
	const	int					fs,			// the length of the filter
			int				   *dbins_t,    // permuted and filtered components of input signal are binned here
	const   unsigned            B,          // number of bins
	const   int                 dH_sig_i)    // modular inverse of sigma and n
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	CUDA_WARN( cudaMemset(dbins_t, 0, B) );

	int T = fs / B;
	int R = fs % B;

	if (n < pow(2, 27))
	{
		PFT_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dbins_t, dx, dfilt_t, n, B, dH_sig_i, T, R);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to launch PFT_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		PFK_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dbins_t, dx, dfilt_t, n, B, dH_sig_i, fs);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to launch PFK_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
			exit(EXIT_FAILURE);
		}
	}
}


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
		const	int				sigma)
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int *dV = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dV, n*sizeof(int)) );

	CUDA_WARN( cudaMemset(dV, 0, n) );

	int *dJ_2sig = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dJ_2sig, B_t*sizeof(int)) );

	makedJ2sig_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dJ_2sig, dJ2, B_t, sigma, W);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch RH_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}	

	RH_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dI, dJ, dV, dJ_2sig, L_t, IF, B_t);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch RH_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	CUDA_WARN( cudaFree(dV) );

	CUDA_WARN( cudaFree(dJ_2sig) );
}


int *gpu_eval(
		const	int			   *dI, 
		const	int			    hIF, 
		const	int			   *dbins_f, 
		const	int			   *dfilter_f, 
		const	unsigned		B, 
		const	unsigned		n, 
		const	unsigned		L, 
		const	int			   *dH_sig)
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int *dx_hat = nullptr;
	CUDA_WARN( cudaMalloc((void **)&dx_hat, n*sizeof(int)) );

	EV_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx_hat, dI, hIF, dbins_f, L, n, dH_sig, B, dfilter_f);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch EV_kernel (Error: " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	int *hx_hat = new int[hIF];
	CUDA_WARN( cudaMemcpy(hx_hat, dx_hat, hIF*sizeof(int), cudaMemcpyDeviceToHost) );

	return hx_hat;
}
