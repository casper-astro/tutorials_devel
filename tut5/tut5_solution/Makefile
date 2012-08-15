# Makefile for CASPER Tutorial 5: Heterogeneous Instrumentation
#
# Created by Jayanth Chennamangalam

# NVCC compiler and flags
NVCC = nvcc
NVFLAGS = --ptxas-options=-v --compiler-bindir=/usr/bin/g77 -O3

# linker options
LFLAGS_CUFFT = -lcufft
LFLAGS_PGPLOT = -L/usr/lib64/pgplot -lpgplot -lcpgplot -lX11
LFLAGS_MATH = -lm

# bin directory
BINDIR = ./bin

all: tut5_fileread.o tut5_kernels.o tut5_plot.o tut5 tags

tut5_fileread.o: tut5_fileread.cu tut5_fileread.h
	$(NVCC) $(NVFLAGS) -c $<

tut5_kernels.o: tut5_kernels.cu tut5_kernels.h
	$(NVCC) $(NVFLAGS) -c $<

tut5_plot.o: tut5_plot.cu tut5_plot.h
	$(NVCC) $(NVFLAGS) -c $<

tut5: tut5_main.cu tut5_main.h tut5_fileread.o tut5_kernels.o tut5_plot.o
# NVCC bug: The .cu file should go at the end, after all .o files, to prevent
#           the 'null character(s) ignored' warning and the subsequent
#           'unrecognized token' errors.
	$(NVCC) $(NVFLAGS) tut5_*.o $<                                            \
	    $(LFLAGS_MATH) $(LFLAGS_CUFFT) $(LFLAGS_PGPLOT) -o $(BINDIR)/$@

tags: *.cu *.h
	ctags --language-force=C *.cu *.h

clean:
	rm *.o

