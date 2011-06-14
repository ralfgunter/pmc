NVCC_FLAGS = -g -DNDEBUG -Xcompiler -pthread -Xcompiler -fno-strict-aliasing -Xcompiler -fwrapv -Xcompiler -O2 -Xcompiler -Wall -Xcompiler -fPIC -I.. -I/usr/local/include/python3.2m
LINK_FLAGS = -shared -Xlinker -export-dynamic -L/usr/local/cuda/lib64 -lcudart -lm -lpthread -ldl -lutil -lpython3.2m

all: pypmc.so

pypmc.o: pypmc.cu
	nvcc $(NVCC_FLAGS) -c pypmc.cu -o pypmc.o

pypmc.so: pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o ../obj/bitset2d.o
	gcc $(LINK_FLAGS) pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o ../obj/bitset2d.o -o $@

clean:
	rm -f pypmc.o pypmc.so
