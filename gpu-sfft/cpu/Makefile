CC = g++
CXXFLAGS = -std=c++14
FFTLIB = -lfftw3 -lm


make: main.cpp AlgoParams.o Algo.o
	$(CC) $(CXXFLAGS) -o gpuSfft  main.cpp AlgoParams.o Algo.o $(FFTLIB)

debug: main.cpp dAlgoParams.o dAlgo.o
	$(CC) $(CXXFLAGS) -g -o debug_gpuSfft main.cpp dAlgoParams.o dAlgo.o $(FFTLIB)

Algo.o: Algo.cpp Algo.hpp
	$(CC) $(CXXFLAGS) -c Algo.cpp

AlgoParams.o: ../utils/AlgoParams.cpp ../utils/AlgoParams.hpp
	$(CC) $(CXXFLAGS) -c ../utils/AlgoParams.cpp

dAlgo.o: Algo.cpp Algo.hpp
	$(CC) $(CXXFLAGS) -o dAlgo.o -c -g Algo.cpp

dAlgoParams.o: ../utils/AlgoParams.cpp ../utils/AlgoParams.hpp
	$(CC) $(CXXFLAGS) -o dAlgoParams.o -c -g ../utils/AlgoParams.cpp

clean:
	rm *.o gpuSfft debug_gpuSfft
