CC = nvcc

NCCL_INC ?= /home/scr2448/piccl/msccl/build/include
NCCL_LIB ?= /home/scr2448/piccl/msccl/build/lib

INC = -I./include -I$(NCCL_INC) -I/usr/lib/x86_64-linux-gnu/openmpi/include
LNK =  -L$(NCCL_LIB)  -lmpi -lnccl -lm

mtrsk: build/main.o build/matryoshka_util.o build/helpers.o
	$(CC) $(LNK) $^ -o $@ $(FLAG)

build/%.o: src/%.c | subdirs
	$(CC) $(INC) -c $^ -o $@ $(FLAG)

.PHONY: clean subdirs

clean:
	rm -f mtrsk build/*.o

subdirs:
	mkdir -p build
