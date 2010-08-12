#include <stdio.h>
#include <cufft.h>
#include <cuda.h>

const char * datafilename = "data/channel256_out";
const char * spectrumfilename = "data/channel256_spectrum";
const int fftlen = 2048;

int main()
{

    FILE *datafile = fopen(datafilename,"r");
    int next_real,next_imaginary; 
    cufftComplex *roachdata;
    cufftComplex *gpudata;
    cufftComplex *gpuspectrum;
    cufftComplex *cpuspectrum;
    
    cudaMallocHost(&roachdata, sizeof(cufftComplex)*fftlen);
    cudaMalloc(&gpudata, sizeof(cufftComplex)*fftlen);
    cudaMalloc(&gpuspectrum, sizeof(cufftComplex)*fftlen);
    cudaMallocHost(&cpuspectrum, sizeof(cufftComplex)*fftlen);
    
    for(int i=0; i<fftlen && fscanf(datafile, "%d %d\n", &next_real, &next_imaginary) != EOF;i++)
    {
        roachdata[i].x = next_real;
        roachdata[i].y = next_imaginary;
    }
    
    
    // allocate device memory for the fft
    cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*fftlen);
    cudaMalloc((void**)&gpuspectrum,sizeof(cufftComplex)*fftlen);

    static cufftHandle plan;
    cufftPlan1d(&plan,fftlen,CUFFT_C2C, 1);


    // allocate device memory and copy over data
    cudaMemcpy(gpudata, roachdata, sizeof(cufftComplex)*fftlen, cudaMemcpyHostToDevice);
    
    // run the fft
    cufftExecC2C(plan,gpudata,gpuspectrum,CUFFT_FORWARD);
    // copy the result back
    cudaMemcpy(cpuspectrum, gpuspectrum, sizeof(cufftComplex)*fftlen, cudaMemcpyDeviceToHost);

    FILE *spectrumfile = fopen(spectrumfilename,"w");
    
    for(int i=0; i<fftlen; i++)
    {
        fprintf(spectrumfile, "%f %f\n", cpuspectrum[i].x, cpuspectrum[i].y);
    }
    
    cufftDestroy(plan);
    cudaFreeHost(roachdata);
    cudaFree(gpudata);
    cudaFree(gpuspectrum);
    cudaFreeHost(cpuspectrum);
}