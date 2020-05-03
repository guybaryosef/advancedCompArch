
# Final Project

Replicating the GPU-SFFT algorithm from the paper: *GPU-SFFT- A  GPU based parallel algorithm for computing the Sparse Fast Fourier Transform(SFFT)* using Nvidia's CUDA.


## How to Install

### FFTW
For the CPU implementation, we use a FFT library called FFTW. To install it, you need to execute:

```sh
wget fftw.org/fftw-3.3.8.tar.gz

tar -zxf fftw-3.3.8.tar.gz
cd fftw-3.3.8
./configure
make
sudo make install
```

### CUDA
For the GPU implementation, we need the CUDA toolkit installed, in particular the nvcc compiler, cuFFT and thrust libraries.


## How to Run

For the CPU version, in the cpu directory
```sh
make
./gpuSfft ../inputs/small_input.txt ../outputs/small_output.txt 

```