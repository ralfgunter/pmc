.PHONY: python2 python3

OS = $(shell uname -s)

SRC = $(wildcard *.cu)
OBJ = $(SRC:%.cu=obj/%.o)

LINK_FLAGS = -lm
NVCC_FLAGS = -Xcompiler -fPIC

# OS X idiosyncrasy: nvcc seems to default to 32-bits,
# while gcc produces 64-bits binaries 
ifeq ($(OS), Darwin)
NVCC_FLAGS += -m64
LINK_FLAGS += -m64
endif


all: opt

obj/%.o: %.cu
	nvcc $(NVCC_FLAGS) -c $< -o $@

pmc: $(OBJ)
	nvcc $(LINK_FLAGS) $(OBJ) -o pmc 

debug: NVCC_FLAGS += -DDEBUG -g -G -O0 -Xptxas="-v"
debug: LINK_FLAGS += -g -G
debug: pmc 

opt: NVCC_FLAGS += -use_fast_math -arch=compute_11
#opt: NVCC_FLAGS += -DNO_FLUENCE
#opt: NVCC_FLAGS += -DNO_MOMENTUM_TRANSFER
#opt: NVCC_FLAGS += -DNO_PATH_LENGTH
opt: pmc 

python3: opt
	@$(MAKE) python3 -C python

python2: opt
	@$(MAKE) python2 -C python

python: python3

clean:
	rm -rf obj/* pmc
	@$(MAKE) clean -C python
