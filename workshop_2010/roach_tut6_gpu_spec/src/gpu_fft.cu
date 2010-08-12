#include <stdio.h>
#include <cufft.h>
#include <cuda.h>

const char * datafilename = "data/channel256_out";
const char * spectrumfilename = "data/channel256_spectrum";
const int fftlen = 2048;

int main()
{
    //open the file containing our recorded data
    FILE *datafile = fopen(datafilename,"r");
    int next_real,next_imaginary; 
    
    //allocate space for the time domain and ffted data on the cpu and the gpu
    cufftComplex *roachdata;
    cufftComplex *gpudata;
    cufftComplex *gpuspectrum;
    cufftComplex *cpuspectrum;
    
    cudaMallocHost(&roachdata, sizeof(cufftComplex)*fftlen);
    cudaMalloc(&gpudata, sizeof(cufftComplex)*fftlen);
    cudaMalloc(&gpuspectrum, sizeof(cufftComplex)*fftlen);
    cudaMallocHost(&cpuspectrum, sizeof(cufftComplex)*fftlen);
    
    //read in the time domain data from the file
    for(int i=0; i<fftlen && fscanf(datafile, "%d %d\n", &next_real, &next_imaginary) != EOF;i++)
    {
        roachdata[i].x = next_real;
        roachdata[i].y = next_imaginary;
    }
    
    //create an fft plan
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