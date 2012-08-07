/** 
 * @file tut5_kernels.cu
 * CASPER Workshop 2011 Tutorial 5: Heterogeneous Instrumentation
 *  CUDA Kernels
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#include "tut5_kernels.h"

extern cufftHandle g_stPlan;
extern float4* g_pf4FFTIn_d;
extern float4* g_pf4FFTOut_d;

/* function that performs the PFB */
__global__ void DoPFB(char4 *pc4Data,
                      float4 *pf4FFTIn,
                      float *pfPFBCoeff)
{
    /*************************************************************************/
    /* Task C: Fill in the PFB code */

    /*************************************************************************/

    return;
}

__global__ void CopyDataForFFT(char4 *pc4Data,
                               float4 *pf4FFTIn)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    pf4FFTIn[i].x = (float) pc4Data[i].x;
    pf4FFTIn[i].y = (float) pc4Data[i].y;
    pf4FFTIn[i].z = (float) pc4Data[i].z;
    pf4FFTIn[i].w = (float) pc4Data[i].w;

    return;
}

/* function that performs the FFT - not a kernel, just a wrapper to an
   API call */
int DoFFT()
{
    /*************************************************************************/
    /* Task D: Call the cufftExecC2C() function to perform FFT */

    /*************************************************************************/

    return EXIT_SUCCESS;
}

__global__ void Accumulate(float4 *pf4FFTOut,
                           float4 *pf4SumStokes)
{
    /*************************************************************************/
    /* Task E: Fill in the accumulation code */

    /*************************************************************************/

    return;
}

