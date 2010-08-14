#include <stdio.h>
#include <cufft.h>
#include <cuda.h>
#include <unistd.h>

const char * datafilenametemplate = "data/channel%d_out";
const char * spectrumfilenametemplate = "data/channel%d_spectrum";
const int filenamebufferlen = 1000;

// size of the fft (should be the same size as the capture brams on the roach)
const int fftlen = 2048;

int main(int argc, char **argv)
{
    // read in command line options
    int channelid=0;
    int ch;

    while ((ch = getopt(argc, argv, "c:h")) != -1) {
        switch (ch) {
            // channel id specified on the command line
            case 'c':
                channelid=atoi(optarg);
                break;
            case 'h':
            default:
                fprintf(stderr,"Usage: gpu_fft -c <channel_id>\n");
                exit(1);
        }
    }
    
    // create the filename strings once the channel id is read in
    char datafilename[filenamebufferlen], spectrumfilename[filenamebufferlen];
    snprintf(datafilename, filenamebufferlen, datafilenametemplate, channelid);
    snprintf(spectrumfilename, filenamebufferlen, spectrumfilenametemplate, channelid);

    // open the file containing our recorded data
    FILE *datafile = fopen(datafilename,"r");
    // check if file was successfully opened
    if(datafile==NULL)
    {
        fprintf(stderr,"Error opening data file for channel %d\n", channelid);
        perror(NULL);
        exit(1);
    }
    
    // allocate space for the time domain and ffted data on the cpu and the gpu
    cufftComplex *roachdata;    // data recorded from the roach on the cpu
    cufftComplex *gpudata;      // data recorded from the roach on the gpu
    cufftComplex *gpuspectrum;  // ffted spectrum of the channel on the gpu
    cufftComplex *cpuspectrum;  // ffted spectrum of the channel on the cpu
    
    // data on the cpu that needs to be transfered to/from the gpu should be allocated with
    // cudaMallocHost rather than a normal malloc call
    cudaMallocHost(&roachdata, sizeof(cufftComplex)*fftlen);
    // data on the gpu must be allocated using cudaMalloc
    cudaMalloc(&gpudata, sizeof(cufftComplex)*fftlen);
    cudaMalloc(&gpuspectrum, sizeof(cufftComplex)*fftlen);
    cudaMallocHost(&cpuspectrum, sizeof(cufftComplex)*fftlen);
    
    //read in the time domain data from the file
    int next_real,next_imaginary; 
    int index;
    for(index=0; index<fftlen && fscanf(datafile, "%d %d\n", &next_real, &next_imaginary) != EOF;index++)
    {
        roachdata[index].x = next_real;
        roachdata[index].y = next_imaginary;
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

    // open the file used to store the ffted result
    FILE *spectrumfile = fopen(spectrumfilename,"w");
    // check if file was successfully opened
    if(spectrumfile==NULL)
    {
        fprintf(stderr,"Error opening output file for channel %d\n", channelid);
        perror(NULL);
        exit(1);
    }
    
    // write the spectrum out to a file
    for(index=0; index<fftlen; index++)
    {
        fprintf(spectrumfile, "%f %f\n", cpuspectrum[index].x, cpuspectrum[index].y);
    }
    
    // deallocate the fft plan
    cufftDestroy(plan);
    
    // deallocate all malloc'ed memory
    cudaFreeHost(roachdata);
    cudaFree(gpudata);
    cudaFree(gpuspectrum);
    cudaFreeHost(cpuspectrum);
}