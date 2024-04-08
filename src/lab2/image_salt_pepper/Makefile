
CXX = icpx

CXXFLAGS=-fsycl -O3

LIBS=


main: main.cpp kernels.cpp 
	$(CXX) $^ $(CXXFLAGS) $(LIBS) -o $@


clean:
	rm -f main
