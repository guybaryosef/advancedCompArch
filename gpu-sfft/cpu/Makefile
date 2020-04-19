CC = g++
CXXFLAGS = -std=c++14

make: main.cpp AlgoParams.o Algo.o
	$(CC) $(CXXFLAGS) -o gpuSfft  main.cpp AlgoParams.o Algo.o

debug: main.cpp AlgoParams.o Algo.o
	$(CC) $(CXXFLAGS) -g -o debug_gpuSfft main.cpp AlgoParams.o Algo.o


Algo.o: Algo.cpp Algo.hpp
	$(CC) $(CXXFLAGS) -c Algo.cpp

AlgoParams.o: ../utils/AlgoParams.cpp ../utils/AlgoParams.hpp
	$(CC) $(CXXFLAGS) -c ../utils/AlgoParams.cpp

clean:
	rm *.o gpuSfft debug_gpuSfft
