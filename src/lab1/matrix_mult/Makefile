CXX = g++
CC = gcc
NVCC = nvcc

CFLAGS=-O3 -I. 
CXXFLAGS=-O3 -I.

LIBS = -lm

SRC =  matrix_mul.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

image: $(SRC) 
	$(CXX) -o matrix_mult  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o matrix_mult

