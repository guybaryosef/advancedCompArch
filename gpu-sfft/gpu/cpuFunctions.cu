

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
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

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
	err = cudaMalloc((void **)&dx, input_size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector x (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dx, hx_ptr, input_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy vector x from host to device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }
  
	int *dfilter_t = nullptr;
	err = cudaMalloc((void **)&dfilter_t, fs*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector filter_t (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(dfilter_t, hfilter_t_ptr, fs*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy vector filter_t from host to device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }
	
	int *dfilter_f = nullptr;
	err = cudaMalloc((void **)&dfilter_f, fs*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector filter_f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(dfilter_f, hfilter_f_ptr, fs*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy vector filter_f from host to device (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

	int *dbins_t = nullptr;
	err = cudaMalloc((void **)&dbins_t, B*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector bins_t_x (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

	int *dbins_f = nullptr;
	err = cudaMalloc((void **)&dbins_f, B*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector bins_f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }
    
	int *dI = nullptr;
	err = cudaMalloc((void **)&dI, input_size);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector I (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

	int *dJ2 = nullptr;
	err = cudaMalloc((void **)&dJ2, B_t*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector J_2 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

	int *dH_sig = nullptr;
	err = cudaMallocManaged((void **)&dH_sig, L*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector H_sig (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
    }

	// executing the algorithm
    for (int i=0; i<L_c; ++i)
        gpu_locLargeCoefGPU(dx, B_t, hx.size(), W, dJ2);

	int *IF;
	err = cudaMallocManaged(&IF, 2*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate unified memory integer IF (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	IF[0] = 0;
	
    for (int i=0; i<L; ++i)
    {
		int sigma = rand() % hx.size();
		std::cout << "i: " << i << std::endl;
        dH_sig[i] = modInverse(sigma, hx.size());

        int *dJ = nullptr;
		err = cudaMalloc((void **)&dJ, B_t*sizeof(int));
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to allocate device vector J (error code " << cudaGetErrorString(err) << ")!" << std::endl;
			exit(EXIT_FAILURE);
		}

        gpu_permFilter(dx, hx.size(), dfilter_t, fs, dbins_t, B, dH_sig[i]);
        gpu_fftCutoff(dJ, dbins_t, dbins_f, B, B_t);

        if (i < L_l)
				gpu_revHash(dI, dJ, B_t, B, hx.size(), L_t, dJ2, W, IF, sigma);
				
		err = cudaFree(dJ);
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to free device vector J (error code " << cudaGetErrorString(err) << ")!" << std::endl;
			exit(EXIT_FAILURE);
		}
    }

    int *output = gpu_eval(dI, IF, dbins_f, dfilter_f, B, hx.size(), L, dH_sig);

	// freeing all the GPU memory
	err = cudaFree(dx);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector x (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
    
	err = cudaFree(dbins_t);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector bins_t_x (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(dbins_f);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector bins_f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(dI);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector I (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dJ2);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector J2 (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(dH_sig);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free uniform memory vector H_sig (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(IF);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free uniform memory integer IF (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	err = cudaFree(dfilter_f);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector filter_f (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dfilter_t);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector filter_t (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

    return std::vector<int>(output, output+IF[0]);
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
	err = cudaMalloc((void **)&dx_prime, W*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector x_prime (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	int *dy_hat = nullptr;
	err = cudaMalloc((void **)&dy_hat, W*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector y_hat (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	int sigma = n/W;
	int tau   = rand() % sigma;
	LLC_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx_prime, dx, W, tau, sigma);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch LLC_kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	gpu_fftCutoff(dJ2, dx_prime, dy_hat, W, B_t);

	err = cudaFree(dy_hat);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector y_hat (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dx_prime);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector x_prime (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
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
	err = cudaMalloc((void**)&dft_output, sizeof(cufftComplex)*dft_length);
	if (cudaGetLastError() != cudaSuccess)
	{
		std::cerr << "Cuda error: Failed to allocate dft output for cuFFT." << std::endl;
		return;	
	}

	cufftReal *dft_input;
	err = cudaMalloc((void**)&dft_input, sizeof(cufftReal)*n_bins);
	if (cudaGetLastError() != cudaSuccess)
	{
		std::cerr << "Cuda error: Failed to allocate dft input for cuFFT." << std::endl;
		return;	
	}

	intToCureal_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dft_input, dinvec_t, n_bins);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch intToCureal_kernel (error code " << cudaGetErrorString(err) << ")!\n";
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

	if (cudaDeviceSynchronize() != cudaSuccess)
	{
		std::cerr << "Cuda error: Failed to synchronize." << std::endl;
		return;	
	}

	curealToInt_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dft_output, dinvec_f, n_bins);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch curealToInt (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dft_output);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector dft_output (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dft_input);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector dft_input (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

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
	err = cudaMalloc((void **)&dsamples_s, m*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector samples_s (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	int *dsamples_I = nullptr;
	err = cudaMalloc((void **)&dsamples_I, m*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector samples_I (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	S_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_yhat, dsamples_s, dsamples_I, m);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch S_kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

    thrust::device_ptr<int> dsamples_s_thrust = thrust::device_pointer_cast(dsamples_s);
	thrust::sort(dsamples_s_thrust, dsamples_s_thrust+m);
	dsamples_s = thrust::raw_pointer_cast(dsamples_s_thrust);

	int cutoff = dsamples_s_thrust[m-B_t-1];


	int *id = nullptr;
	err = cudaMalloc((void **)&id, sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device int id (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	C_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dId, cutoff, dsamples_I, m, id, B_t);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch C_kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	thrust::device_ptr<int> dId_thrust = thrust::device_pointer_cast(dId);
	thrust::sort(dId_thrust, dId_thrust+B_t);
	dId = thrust::raw_pointer_cast(dId_thrust);

	err = cudaFree(dsamples_s);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector samples_s (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dsamples_I);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector samples_I (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
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

	err = cudaMemset(dbins_t, 0, B);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to set memory of device vector bins_t (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	int T = fs / B;
	int R = fs % B;

	if (n < pow(2, 27))
	{
		PFT_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dbins_t, dx, dfilt_t, n, B, dH_sig_i, T, R);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to launch PFT_kernel (error code " << cudaGetErrorString(err) << ")!\n";
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		PFK_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dbins_t, dx, dfilt_t, n, B, dH_sig_i, fs);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "Failed to launch PFK_kernel (error code " << cudaGetErrorString(err) << ")!\n";
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
	err = cudaMalloc((void **)&dV, n*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector V (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}	

	err = cudaMemset(dV, 0, n);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to set memory of device vector V (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	int *dJ_2sig = nullptr;
	err = cudaMalloc((void **)&dJ_2sig, B_t*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector J_2sig (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}	

	for (int i=0; i<B_t; ++i)
		dJ_2sig[i] = (dJ2[i]*sigma) % W;

	RH_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dI, dJ, dV, dJ_2sig, L_t, IF, B_t);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch RH_kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dV);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector V (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	err = cudaFree(dJ_2sig);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to free device vector J_2sig (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
}


int *gpu_eval(
		const	int			   *dI, 
		const	int			   *IF, 
		const	int			   *dbins_f, 
		const	int			   *dfilter_f, 
		const	unsigned		B, 
		const	unsigned		n, 
		const	unsigned		L, 
		const	int			   *dH_sig)
{
	cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls

	int *dx_hat = nullptr;
	err = cudaMalloc((void **)&dx_hat, n*sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to allocate device vector x_hat (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	EV_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dx_hat, dI, IF, dbins_f, L, n, dH_sig, B, dfilter_f);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to launch EV_kernel (error code " << cudaGetErrorString(err) << ")!\n";
		exit(EXIT_FAILURE);
	}

	int *hx_hat = new int[IF[0]];
	err = cudaMemcpy(hx_hat, dx_hat, IF[0]*sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to copy vector x_hat from device to host (error code " << cudaGetErrorString(err) << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	return hx_hat;
}
