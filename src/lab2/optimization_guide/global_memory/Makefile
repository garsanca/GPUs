CXX=icpx
CXXFLAGS=-fsycl -I. -O3 -gline-tables-only -fdebug-info-for-profiling
LIBS=

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: matrix_add.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
