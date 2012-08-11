/**
 * @file tut5_main.h
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Top-level header file
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#ifndef __TUT5_MAIN_H__
#define __TUT5_MAIN_H__

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>

#include <string.h>     /* for memset(), strncpy(), memcpy(), strerror() */
#include <sys/types.h>  /* for open() */
#include <sys/stat.h>   /* for open() */
#include <fcntl.h>      /* for open() */
#include <unistd.h>     /* for close() and usleep() */
#include <cpgplot.h>    /* for cpg*() */
#include <float.h>      /* for FLT_MAX */
#include <getopt.h>     /* for option parsing */
#include <assert.h>     /* for assert() */
#include <errno.h>      /* for errno */
#include <signal.h>     /* for signal-handling */
#include <math.h>       /* for log10f() in Plot() */
#include <sys/time.h>   /* for gettimeofday() */

#define FALSE               0
#define TRUE                1

#define LEN_GENSTRING       256

#define DEF_PFB_ON          FALSE

#define NUM_BYTES_PER_SAMP  4
#define DEF_LEN_SPEC        1024        /* default value for g_iNFFT */

#define DEF_SIZE_READ       33554432    /* 32 MB - block size in VEGAS input
                                           buffer */
#define LEN_DATA            (NUM_BYTES_PER_SAMP * g_iNFFT)

#define DEF_ACC             1           /* default number of spectra to
                                           accumulate */
/* for PFB */
#define NUM_TAPS            8       /* number of multiples of g_iNFFT */
#define FILE_COEFF_PREFIX   "coeff"
#define FILE_COEFF_DATATYPE "float"
#define FILE_COEFF_SUFFIX   ".dat"

#define DEF_NUM_SUBBANDS    1

#define FFTPLAN_RANK        1
#define FFTPLAN_ISTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_OSTRIDE     (2 * g_iNumSubBands)
#define FFTPLAN_IDIST       1
#define FFTPLAN_ODIST       1
#define FFTPLAN_BATCH       (2 * g_iNumSubBands)

#define USEC2SEC            1e-6

/* PGPLOT macro definitions */
#define PG_DEV              "1/XS"
#define PG_VP_ML            0.10    /* left margin */
#define PG_VP_MR            0.90    /* right margin */
#define PG_VP_MB            0.12    /* bottom margin */
#define PG_VP_MT            0.98    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11

typedef unsigned char BYTE;

/**
 * Initialises the program.
 */
int Init(void);

/**
 * Reads all data from the input file and loads it into memory.
 */
int LoadDataToMem(void);

/**
 * Reads one block (32MB) of data form memory.
 */
int ReadData(void);

/*
 * Perform polyphase filtering.
 *
 * @param[in]   pc4Data     Input data (raw data read from memory)
 * @param[out]  pf4FFTIn    Output data (input to FFT)
 */
__global__ void DoPFB(char4* pc4Data,
                      float4* pf4FFTIn,
                      float* pfPFBCoeff);
__global__ void CopyDataForFFT(char4* pc4Data,
                               float4* pf4FFTIn);
int DoFFT(void);
__global__ void Accumulate(float4 *pf4FFTOut,
                           float4* pfSumStokes);
void CleanUp(void);

#define CUDASafeCallWithCleanUp(iRet)   __CUDASafeCallWithCleanUp(iRet,       \
                                                                  __FILE__,   \
                                                                  __LINE__,   \
                                                                  &CleanUp)

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void));

#if BENCHMARKING
void PrintBenchmarks(float fAvgPFB,
                     int iCountPFB,
                     float fAvgCpInFFt,
                     int iCountCpInFFT,
                     float fAvgFFT,
                     int iCountFFT,
                     float fAvgAccum,
                     int iCountAccum,
                     float fAvgCpOut,
                     int iCountCpOut);
#endif

/* PGPLOT function declarations */
int InitPlot(void);
void Plot(void);

int RegisterSignalHandlers();
void HandleStopSignals(int iSigNo);
void PrintUsage(const char* pcProgName);

#endif  /* __TUT5_MAIN_H__ */

