.PHONY: python

DEV_SRC  = $(wildcard *.cu)
DEV_OBJ  = $(DEV_SRC:%.cu=obj/%.o)
HOST_SRC = $(wildcard *.cc)
HOST_OBJ = $(HOST_SRC:%.cc=obj/%.o)

LINK_FLAGS = -lm
NVCC_FLAGS = -Xcompiler -fPIC -I /home/ralfgunter/NVIDIA_GPU_Computing_SDK/C/common/inc
GCC_FLAGS  = -fPIC

all: opt

obj/%.o: %.cu
	nvcc $(NVCC_FLAGS) -c $< -o $@

obj/%.o: %.cc
	g++ $(GCC_FLAGS) -c $< -o $@

pmc: $(DEV_OBJ) $(HOST_OBJ)
	nvcc $(LINK_FLAGS) $(DEV_OBJ) $(HOST_OBJ) -o pmc 

debug: NVCC_FLAGS += -g -G -use_fast_math -Xptxas="-v"
debug: LINK_FLAGS += -g -G
debug: GCC_FLAGS += -g
debug: pmc 

opt: NVCC_FLAGS += -use_fast_math -arch=compute_11
opt: pmc 

python: opt
	@$(MAKE) -C python

clean:
	rm -rf obj/* pmc
	@$(MAKE) clean -C python
