CXX = dpcpp
COMPFLAGS = -g -std=c++11 
OPTFLAGS = -O2
INCLUDES = 
LIBS=-L/usr/lib/gcc/x86_64-linux-gnu/11


CXXFLAGS = $(COMPFLAGS) $(OPTFLAGS) 

SOURCES = GSimulation.cpp main.cpp

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: clean cpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

cpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC) $(LIBS)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x  

#----------------------------------------------------------------



