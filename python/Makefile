OS = $(shell uname -s)
ARCH = $(shell uname -m)

# Default is to compile against python 3.x
all: python3
PY_VERSION = 3

# Python 3.2
P3_NUMPY_INC = -I$(shell python3.2 -c "import numpy; print(numpy.get_include())") 
P3_INCS = $(shell python3.2-config --includes) $(P3_NUMPY_INC)
P3_LINK = $(shell python3.2-config --ldflags)
# OS X idiosyncrasy
ifeq ($(OS), Darwin)
P3_LINK = -ldl -framework CoreFoundation -lpython3.2 -L/Library/Frameworks/Python.framework/Versions/3.2/lib/python3.2/config-3.2m
endif

python3: PY_INCS = $(P3_INCS)
python3: PY_LINK = $(P3_LINK)
python3: PY_VERSION = 3
python3: pypmc.so

# Python 2
P2_NUMPY_INC = -I$(shell python -c "import numpy; print(numpy.get_include())") 
P2_INCS = $(shell python-config --includes) $(P2_NUMPY_INC)
P2_LINK = $(shell python-config --ldflags)
# OS X idiosyncrasy
ifeq ($(OS), Darwin)
P2_LINK = -ldl -framework CoreFoundation -lpython2.7 -L/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config
endif

python2: PY_INCS = $(P2_INCS)
python2: PY_LINK = $(P2_LINK)
python2: PY_VERSION = 2
python2: pypmc.so

# Common flags
ifeq ($(ARCH), x86_64)
CUDA_ARCH = 64
endif

NVCC_FLAGS = -g -DNDEBUG -Xcompiler -pthread -Xcompiler -fno-strict-aliasing -Xcompiler -fwrapv -Xcompiler -O2 -Xcompiler -Wall -Xcompiler -fPIC -I.. $(PY_INCS) -DPYTHON=$(PY_VERSION)
CUDA_LINK += -L/usr/local/cuda/lib$(CUDA_ARCH) -lcudart
LINK_FLAGS = -shared $(CUDA_LINK) $(PY_LINK)
ifeq ($(OS), Darwin)
LINK_FLAGS += -m64
NVCC_FLAGS += -m64
CUDA_LINK += -m64
endif

pypmc.o: pypmc.cu
	nvcc $(NVCC_FLAGS) -c pypmc.cu -o pypmc.o

pypmc.so: pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o
	g++ $(LINK_FLAGS) pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o -o $@

clean:
	rm -f pypmc.o pypmc.so
