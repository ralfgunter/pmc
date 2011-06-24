# Default is to compile against python 3.x
all: python3
PY_VERSION = 3

# Python 3
P3_LINK = -lpthread -ldl -lutil -lm -lpython3.2mu
P3_INCS = -I/usr/include/python3.2mu

python3: PY_INCS = $(P3_INCS)
python3: PY_LINK = $(P3_LINK)
python3: PY_VERSION = 3
python3: pypmc.so

# Python 2
P2_LINK = -lpthread -ldl -lutil -lm -lpython2.7
P2_INCS = -I/usr/include/python2.7

python2: PY_INCS = $(P2_INCS)
python2: PY_LINK = $(P2_LINK)
python2: PY_VERSION = 2
python2: pypmc.so

# Common flags
NVCC_FLAGS = -g -DPYTHON=$(PY_VERSION) -DNDEBUG -Xcompiler -pthread -Xcompiler -fno-strict-aliasing -Xcompiler -fwrapv -Xcompiler -O2 -Xcompiler -Wall -Xcompiler -fPIC -I.. $(PY_INCS)
LINK_FLAGS = -shared -Xlinker -export-dynamic -L/usr/local/cuda/lib64 -lcudart $(PY_LINK)


pypmc.o: pypmc.cu
	nvcc $(NVCC_FLAGS) -c pypmc.cu -o pypmc.o

pypmc.so: pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o ../obj/bitset2d.o
	gcc $(LINK_FLAGS) pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o ../obj/bitset2d.o -o $@

clean:
	rm -f pypmc.o pypmc.so
