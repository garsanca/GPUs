CXX=dpcpp
CXXFLAGS=-I. -O3 -gline-tables-only -fdebug-info-for-profiling
LIBS=-L/usr/lib/gcc/x86_64-linux-gnu/11

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

exec: matrix_add.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS) 

clean:
	rm -f exec *.o
