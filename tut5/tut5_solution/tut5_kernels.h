/**
 * @file tut5_kernels.h
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  CUDA Kernels - Header File
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#ifndef __TUT5_KERNELS_H__
#define __TUT5_KERNELS_H__

#include "tut5_main.h"

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


#endif  /* __TUT5_KERNELS_H__ */

