## Makefile di esempio per il progetto di High Performance Computing
## 2024/2025, corso di laurea in Ingegneria e Scienze Informatiche,
## Universita' di Bologna.

## Ultima modifica: 2024-12-05, Moreno Marzolla <moreno.marzolla(at)unibo.it>

## Questo e' un frammento del Makefile utilizzato negli esempi
## illustrati durante il corso. Questo file puo' essere modificato e
## personalizzato in base alle proprie esigenze. Si puo' anche
## decidere di non usarlo; in tal caso indicare le istruzioni di
## compilazione nel file README presente nella directory a livello
## superiore.
##
## Se si decide di usare questo makefile, il comando "make" dovrebbe
## compilare tutti i programmi consegnati senza errori né warning.  Si
## consiglia pertanto di rimuovere eventuali target non presenti
## nell'archivio consegnato.
##
## Questo Makefile compila i file "omp-*.c" usando il flag -fopenmp, i
## file "cuda-*.cu" con il compilatore nvcc, e i file "mpi-*.c" con
## mpicc.
##
## I principali target definiti da questo makefile sono:
##
## make         compila tutti i sorgenti disponibili
## make clean   cancella i file temporanei e gli eseguibili
## make openmp  compila la versione OpenMP
## make mpi     compila la versione MPI
## make cuda    compila la versione CUDA
## make hip     genera e compila la versione HIP

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE_SERIAL:=skyline
CFLAGS+=-std=c99 -Wall -Wpedantic -Ofast -D_XOPEN_SOURCE=600
LDLIBS+=-lm
NVCC:=nvcc
NVCFLAGS+=-Wno-deprecated-gpu-targets
NVLDLIBS+=-lm
CC:= gcc
EXE_HIP := $(basename $(wildcard hip-*.cu.hip))
HIPCC := hipcc
CUDA_SRC := $(wildcard cuda-*.cu)
HIP_SRC := $(CUDA_SRC:.cu=.cu.hip)
EXE_HIP := hip-$(basename $(CUDA_SRC))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_CUDA) $(EXE_HIP)

.PHONY: clean

ALL: $(EXE)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@ $(NVLDLIBS)

$(EXE_OMP): CFLAGS+=-fopenmp

openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc

mpi: $(EXE_MPI)

cuda: $(EXE_CUDA)

hip: $(EXE_HIP)

%.cu.hip: %.cu
	perl hipify-perl $< > $@

$(EXE_HIP): $(HIP_SRC)
	$(HIPCC) $< -o $@

clean:
	\rm -f $(EXE) *.o *~
