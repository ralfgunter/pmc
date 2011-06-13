NVCC_FLAGS = -g -DNDEBUG -Xcompiler -pthread -Xcompiler -fno-strict-aliasing -Xcompiler -fwrapv -Xcompiler -O2 -Xcompiler -Wall -Xcompiler -fPIC -I /usr/include/python2.7 -I.. 
LINK_FLAGS = -pthread -shared -Wl,-O1 --Wl,-Bsymbolic-functions -L/usr/local/cuda/lib64 -lcudart -lm -stdc++

all: pypmc.so

pypmc.o: pypmc.cu
	nvcc $(NVCC_FLAGS) -c pypmc.cu -o pypmc.o

pypmc.so: pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o
	g++ $(LINK_FLAGS) pypmc.o ../obj/io.o ../obj/mem.o ../obj/kernel.o -o $@

clean:
	rm -f pypmc.o pypmc.so
