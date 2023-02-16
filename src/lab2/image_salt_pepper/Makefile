
CXX = dpcpp

CXXFLAGS=-O3

LIBS=-L/usr/lib/gcc/x86_64-linux-gnu/11


main: main.cpp kernels.cpp 
	$(CXX) $^ $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f main
