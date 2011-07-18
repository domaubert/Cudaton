##########################################
C_LIBS = -lm -lcuda  -L/usr/local/cuda/lib64 # -I/usr/local/cuda/SDK/C/common/inc -L/usr/local/cuda/SDK/C/lib 
CUDA_LIBS =-lcudart -lcutil_x86_64
C_OBJS= Main.o communication.o
CUDA_OBJS= Allocation.o Io.o Boundary.o Explicit.o cosmo.o timestep.o Interface.o 
DEFINES  = -DVERBOSE   -DSSHARP -DTIMINGS  -DNEWTEMP -DS_50000 -DDUMPFLUX -DCOSMO -DFLAT_COSMO -DPEDRO -DCOOLING -DWMPI #-DSDISCRETE -DTEST_STROMGREN -DDUMPFLUX #-DCOOLING -DDUMPGRID # -DWMPI -DTITANE#-DCOOLING #-DCOSMO -DFLAT_COSMO #-DTESTCOOL #-DRAND_SRC#-DSPOTSOURCE #-DSSHARP #-DCOSMO #-DISOTROP # #-DSPLIT #-DISOTROP #-DCPURUN -DCOOLING -DSSHARP 

NVCC= nvcc #--device-emulation
CC = mpicc 
OBJECTS = $(C_OBJS) $(CUDA_OBJS)
EXECUTABLE = cudaton_mpi
.c.o:
	$(CC) $(DEFINES) -c $<
%.o:%.cu
	$(NVCC) $(DEFINES) $(C_LIBS) $(CUDA_LIBS) -c $*.cu	

all:$(C_OBJS) $(CUDA_OBJS)
	$(CC)  $(CUDA_OBJS) $(C_OBJS) $(C_LIBS) $(CUDA_LIBS) -o $(EXECUTABLE)

clean:
	rm -f *.o *.cudafe1.* *.cudafe2.* *.hash *.ptx *fatbin.c *.cubin *.cpp* $(EXECUTABLE) *~
