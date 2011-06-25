.PHONY: python2 python3

OS = $(shell uname -s)

DEV_SRC  = $(wildcard *.cu)
DEV_OBJ  = $(DEV_SRC:%.cu=obj/%.o)
HOST_SRC = $(wildcard *.cc)
HOST_OBJ = $(HOST_SRC:%.cc=obj/%.o)

LINK_FLAGS = -lm
NVCC_FLAGS = -Xcompiler -fPIC
GCC_FLAGS  = -fPIC

# OS X idiosyncrasy: nvcc seems to default to 32-bits,
# while gcc produces 64-bits binaries 
ifeq ($(OS), Darwin)
GCC_FLAGS += -m64
NVCC_FLAGS += -m64
LINK_FLAGS += -m64
endif


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

python3: opt
	@$(MAKE) python3 -C python

python2: opt
	@$(MAKE) python2 -C python

python: python3

clean:
	rm -rf obj/* pmc
	@$(MAKE) clean -C python
