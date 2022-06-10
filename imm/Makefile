CXX = icc
CXXFLAGS = -g -I/apps/cuda/cuda-11.4/include -std=c++11 -Wall -O3
HDRS = species.h plasma.h
OBJS = main.o species.o species_mtcpu.o species_gpu.o plasma.o config.o
LDLIBS = -lblas -llapack -lOpenCL

%.o : %.cpp $(HDRS)
	$(CXX) -fopenmp $(CXXFLAGS) -g -o $@ -c $<

plas: $(OBJS)
	$(CXX) -fopenmp -g -o $@ $^ $(LDLIBS)

all: plas