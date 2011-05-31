DEV_SRC  = $(wildcard *.cu)
DEV_OBJ  = $(DEV_SRC:%.cu=obj/%.o)
HOST_SRC = $(wildcard *.cc)
HOST_OBJ = $(HOST_SRC:%.cc=obj/%.o)

LINK_FLAGS = -m32 -lm
NVCC_FLAGS = -m32
GCC_FLAGS  = -m32

all: tMCimg 

obj/%.o: %.cu
	nvcc $(NVCC_FLAGS) -c $< -o $@

obj/%.o: %.cc
	g++ $(GCC_FLAGS) -c $< -o $@

tMCimg: $(DEV_OBJ) $(HOST_OBJ)
	nvcc $(LINK_FLAGS) $(DEV_OBJ) $(HOST_OBJ) -o tMCimg

debug: NVCC_FLAGS += -g -G -Xptxas="-v"
debug: LINK_FLAGS += -g -G
debug: GCC_FLAGS += -g
debug: tMCimg

clean:
	rm -rf obj/* tMCimg
