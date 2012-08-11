/** 
 * @file tut5.cu
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Top-level file
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#include "tut5_main.h"

/* plotting */
extern float* g_pfSumPowX;
extern float* g_pfSumPowY;
extern float* g_pfSumStokesRe;
extern float* g_pfSumStokesIm;
extern float* g_pfFreq;
extern float g_fFSamp;

int g_iIsDataReadDone = FALSE;
char4* g_pc4InBuf = NULL;
char4* g_pc4InBufRead = NULL;
char4* g_pc4Data_d = NULL;              /* raw data starting address */
char4* g_pc4DataRead_d = NULL;          /* raw data read pointer */
int g_iNFFT = DEF_LEN_SPEC;
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 g_dimBAccum(1, 1, 1);
dim3 g_dimGAccum(1, 1);
float4* g_pf4FFTIn_d = NULL;
float4* g_pf4FFTOut_d = NULL;
cufftHandle g_stPlan = {0};
float4* g_pf4SumStokes = NULL;
float4* g_pf4SumStokes_d = NULL;
int g_iIsPFBOn = DEF_PFB_ON;
int g_iNTaps = 1;                       /* 1 if no PFB, NUM_TAPS if PFB */
/* BUG: crash if file size is less than 32MB */
int g_iSizeRead = DEF_SIZE_READ;
int g_iNumSubBands = DEF_NUM_SUBBANDS;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};
float *g_pfPFBCoeff = NULL;
float *g_pfPFBCoeff_d = NULL;

int main(int argc, char *argv[])
{
    int iRet = EXIT_SUCCESS;
    int iSpecCount = 0;
    int iNumAcc = DEF_ACC;
    int iProcData = 0;
    cudaError_t iCUDARet = cudaSuccess;
    struct timeval stStart = {0};
    struct timeval stStop = {0};
    const char *pcProgName = NULL;
    int iNextOpt = 0;
    /* valid short options */
    const char* const pcOptsShort = "hb:n:pa:s:";
    /* valid long options */
    const struct option stOptsLong[] = {
        { "help",           0, NULL, 'h' },
        { "nsub",           1, NULL, 'b' },
        { "nfft",           1, NULL, 'n' },
        { "pfb",            0, NULL, 'p' },
        { "nacc",           1, NULL, 'a' },
        { "fsamp",          1, NULL, 's' },
        { NULL,             0, NULL, 0   }
    };

    /* get the filename of the program from the argument list */
    pcProgName = argv[0];

    /* parse the input */
    do
    {
        iNextOpt = getopt_long(argc, argv, pcOptsShort, stOptsLong, NULL);
        switch (iNextOpt)
        {
            case 'h':   /* -h or --help */
                /* print usage info and terminate */
                PrintUsage(pcProgName);
                return EXIT_SUCCESS;

            case 'b':   /* -b or --nsub */
                /* set option */
                g_iNumSubBands = (int) atoi(optarg);
                break;

            case 'n':   /* -n or --nfft */
                /* set option */
                g_iNFFT = (int) atoi(optarg);
                break;

            case 'p':   /* -p or --pfb */
                /* set option */
                g_iIsPFBOn = TRUE;
                break;

            case 'a':   /* -a or --nacc */
                /* set option */
                iNumAcc = (int) atoi(optarg);
                break;

            case 's':   /* -s or --fsamp */
                /* set option */
                g_fFSamp = (float) atof(optarg);
                break;

            case '?':   /* user specified an invalid option */
                /* print usage info and terminate with error */
                (void) fprintf(stderr, "ERROR: Invalid option!\n");
                PrintUsage(pcProgName);
                return EXIT_FAILURE;

            case -1:    /* done with options */
                break;

            default:    /* unexpected */
                assert(0);
        }
    } while (iNextOpt != -1);

    /* initialise */
    iRet = Init();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Init failed!\n");
        CleanUp();
        return EXIT_FAILURE;
    }

    (void) gettimeofday(&stStart, NULL);
    while (TRUE)
    {
        if (g_iIsPFBOn)
        {
            /* do pfb */
            DoPFB<<<g_dimGPFB, g_dimBPFB>>>(g_pc4DataRead_d,
                                            g_pf4FFTIn_d,
                                            g_pfPFBCoeff_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                return EXIT_FAILURE;
            }
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }
        else
        {
            CopyDataForFFT<<<g_dimGCopy, g_dimBCopy>>>(g_pc4DataRead_d,
                                                       g_pf4FFTIn_d);
            CUDASafeCallWithCleanUp(cudaThreadSynchronize());
            iCUDARet = cudaGetLastError();
            if (iCUDARet != cudaSuccess)
            {
                (void) fprintf(stderr,
                               "ERROR: File <%s>, Line %d: %s\n",
                               __FILE__,
                               __LINE__,
                               cudaGetErrorString(iCUDARet));
                /* free resources */
                CleanUp();
                return EXIT_FAILURE;
            }
            /* update the data read pointer */
            g_pc4DataRead_d += (g_iNumSubBands * g_iNFFT);
        }

        /* do fft */
        iRet = DoFFT();
        if (iRet != EXIT_SUCCESS)
        {
            (void) fprintf(stderr, "ERROR! FFT failed!\n");
            CleanUp();
            return EXIT_FAILURE;
        }

        /* accumulate power x, power y, stokes, if the blanking bit is
           not set */
        Accumulate<<<g_dimGAccum, g_dimBAccum>>>(g_pf4FFTOut_d,
                                                 g_pf4SumStokes_d);
        CUDASafeCallWithCleanUp(cudaThreadSynchronize());
        iCUDARet = cudaGetLastError();
        if (iCUDARet != cudaSuccess)
        {
            (void) fprintf(stderr,
                           "ERROR: File <%s>, Line %d: %s\n",
                           __FILE__,
                           __LINE__,
                           cudaGetErrorString(iCUDARet));
            /* free resources */
            CleanUp();
            return EXIT_FAILURE;
        }
        ++iSpecCount;
        if (iSpecCount == iNumAcc)
        {
            /* dump to buffer */
            CUDASafeCallWithCleanUp(cudaMemcpy(g_pf4SumStokes,
                                               g_pf4SumStokes_d,
                                               (g_iNumSubBands
                                                * g_iNFFT
                                                * sizeof(float4)),
                                                cudaMemcpyDeviceToHost));

            /* NOTE: Plot() will modify data! */
            Plot();
            (void) usleep(500000);

            /* reset time */
            iSpecCount = 0;
            /* zero accumulators */
            CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                               '\0',
                                               (g_iNumSubBands
                                                * g_iNFFT
                                                * sizeof(float4))));
        }

        /* if time to read from input buffer */
        iProcData += (g_iNumSubBands * g_iNFFT * sizeof(char4));
        if ((g_iSizeRead
             - ((g_iNTaps - 1) * g_iNumSubBands * g_iNFFT * sizeof(char4)))
            == iProcData)
        {
            if (!(g_iIsDataReadDone))
            {
                /* read data from input buffer */
                iRet = ReadData();
                if (iRet != EXIT_SUCCESS)
                {
                    (void) fprintf(stderr, "ERROR: Data reading failed!\n");
                    break;
                }
            }
            else    /* no more data to be read in this file, open next file */
            {
                (void) printf("File done.\n");

                /* load data into memory */
                iRet = LoadDataToMem();
                if (iRet != EXIT_SUCCESS)
                {
                    (void) fprintf(stderr,
                                   "ERROR! Loading to memory failed!\n");
                    return EXIT_FAILURE;
                }

                /* read data from input buffer */
                iRet = ReadData();
                if (iRet != EXIT_SUCCESS)
                {
                    (void) fprintf(stderr, "ERROR: Data reading failed!\n");
                    break;
                }
            }
            iProcData = 0;
        }
    }
    (void) gettimeofday(&stStop, NULL);
    (void) printf("Time taken (barring Init()): %gs\n",
                  ((stStop.tv_sec + (stStop.tv_usec * USEC2SEC))
                   - (stStart.tv_sec + (stStart.tv_usec * USEC2SEC))));

    CleanUp();

    return EXIT_SUCCESS;
}

/* function that creates the FFT plan, allocates memory, initialises counters,
   etc. */
int Init()
{
    int iDevCount = 0;
    cudaDeviceProp stDevProp = {0};
    int iRet = EXIT_SUCCESS;
    cufftResult iCUFFTRet = CUFFT_SUCCESS;
    int iMaxThreadsPerBlock = 0;

    iRet = RegisterSignalHandlers();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Signal-handler registration failed!\n");
        return EXIT_FAILURE;
    }

    /* since CUDASafeCallWithCleanUp() calls cudaGetErrorString(),
       it should not be used here - will cause crash if no CUDA device is
       found */
    (void) cudaGetDeviceCount(&iDevCount);
    if (0 == iDevCount)
    {
        (void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
        return EXIT_FAILURE;
    }

    /* just use the first device */
    CUDASafeCallWithCleanUp(cudaSetDevice(0));

    CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
    iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;

    if (g_iIsPFBOn)
    {
        /* set number of taps to NUM_TAPS if PFB is on, else number of
           taps = 1 */
        g_iNTaps = NUM_TAPS;

        g_pfPFBCoeff = (float *) malloc(g_iNumSubBands
                                        * g_iNTaps
                                        * g_iNFFT
                                        * sizeof(float));
        if (NULL == g_pfPFBCoeff)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }

        /* allocate memory for the filter coefficient array on the device */
        CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pfPFBCoeff_d,
                                           g_iNumSubBands
                                           * g_iNTaps
                                           * g_iNFFT
                                           * sizeof(float)));

        /* read filter coefficients */
        /* build file name */
        (void) sprintf(g_acFileCoeff,
                       "%s_%s_%d_%d_%d%s",
                       FILE_COEFF_PREFIX,
                       FILE_COEFF_DATATYPE,
                       g_iNTaps,
                       g_iNFFT,
                       g_iNumSubBands,
                       FILE_COEFF_SUFFIX);
        g_iFileCoeff = open(g_acFileCoeff, O_RDONLY);
        if (g_iFileCoeff < EXIT_SUCCESS)
        {
            (void) fprintf(stderr,
                           "ERROR: Opening filter coefficients file %s "
                           "failed! %s.\n",
                           g_acFileCoeff,
                           strerror(errno));
            return EXIT_FAILURE;
        }

        iRet = read(g_iFileCoeff,
                    g_pfPFBCoeff,
                    g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float));
        if (iRet != (g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float)))
        {
            (void) fprintf(stderr,
                           "ERROR: Reading filter coefficients failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
        (void) close(g_iFileCoeff);

        /* copy filter coefficients to the device */
        CUDASafeCallWithCleanUp(cudaMemcpy(g_pfPFBCoeff_d,
                   g_pfPFBCoeff,
                   g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float),
                   cudaMemcpyHostToDevice));
    }

    /* allocate memory for data array - 32MB is the block size for the VEGAS
       input buffer */
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc4Data_d, g_iSizeRead));
    g_pc4DataRead_d = g_pc4Data_d;

    /* load data from the first file into memory */
    iRet = LoadDataToMem();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! Loading to memory failed!\n");
        return EXIT_FAILURE;
    }

    /* calculate kernel parameters */
    if (g_iNFFT < iMaxThreadsPerBlock)
    {
        g_dimBPFB.x = g_iNFFT;
        g_dimBCopy.x = g_iNFFT;
        g_dimBAccum.x = g_iNFFT;
    }
    else
    {
        g_dimBPFB.x = iMaxThreadsPerBlock;
        g_dimBCopy.x = iMaxThreadsPerBlock;
        g_dimBAccum.x = iMaxThreadsPerBlock;
    }
    g_dimGPFB.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;
    g_dimGCopy.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;
    g_dimGAccum.x = (g_iNumSubBands * g_iNFFT) / iMaxThreadsPerBlock;

    iRet = ReadData();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Reading data failed!\n");
        return EXIT_FAILURE;
    }

    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTIn_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4FFTOut_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));

    g_pf4SumStokes = (float4 *) malloc(g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4));
    if (NULL == g_pf4SumStokes)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf4SumStokes_d,
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));
    CUDASafeCallWithCleanUp(cudaMemset(g_pf4SumStokes_d,
                                       '\0',
                                       g_iNumSubBands
                                       * g_iNFFT
                                       * sizeof(float4)));

    /* create plan */
    iCUFFTRet = cufftPlanMany(&g_stPlan,
                              FFTPLAN_RANK,
                              &g_iNFFT,
                              &g_iNFFT,
                              FFTPLAN_ISTRIDE,
                              FFTPLAN_IDIST,
                              &g_iNFFT,
                              FFTPLAN_OSTRIDE,
                              FFTPLAN_ODIST,
                              CUFFT_C2C,
                              FFTPLAN_BATCH);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR: Plan creation failed!\n");
        return EXIT_FAILURE;
    }

    iRet = InitPlot();
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Plotting initialisation failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/* function that frees resources */
void CleanUp()
{
    /* free resources */
    if (g_pc4InBuf != NULL)
    {
        free(g_pc4InBuf);
        g_pc4InBuf = NULL;
    }
    if (g_pc4Data_d != NULL)
    {
        (void) cudaFree(g_pc4Data_d);
        g_pc4Data_d = NULL;
    }
    if (g_pf4FFTIn_d != NULL)
    {
        (void) cudaFree(g_pf4FFTIn_d);
        g_pf4FFTIn_d = NULL;
    }
    if (g_pf4FFTOut_d != NULL)
    {
        (void) cudaFree(g_pf4FFTOut_d);
        g_pf4FFTOut_d = NULL;
    }
    if (g_pf4SumStokes != NULL)
    {
        free(g_pf4SumStokes);
        g_pf4SumStokes = NULL;
    }
    if (g_pf4SumStokes_d != NULL)
    {
        (void) cudaFree(g_pf4SumStokes_d);
        g_pf4SumStokes_d = NULL;
    }

    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);

    /* destroy plan */
    /* TODO: check for plan */
    (void) cufftDestroy(g_stPlan);

    if (g_pfSumPowX != NULL)
    {
        free(g_pfSumPowX);
        g_pfSumPowX = NULL;
    }
    if (g_pfSumPowY != NULL)
    {
        free(g_pfSumPowY);
        g_pfSumPowY = NULL;
    }
    if (g_pfSumStokesRe != NULL)
    {
        free(g_pfSumStokesRe);
        g_pfSumStokesRe = NULL;
    }
    if (g_pfSumStokesIm != NULL)
    {
        free(g_pfSumStokesIm);
        g_pfSumStokesIm = NULL;
    }
    if (g_pfFreq != NULL)
    {
        free(g_pfFreq);
        g_pfFreq = NULL;
    }

    /* TODO: check if open */
    cpgclos();

    return;
}

/*
 * Registers handlers for SIGTERM and CTRL+C
 */
int RegisterSignalHandlers()
{
    struct sigaction stSigHandler = {{0}};
    int iRet = EXIT_SUCCESS;

    /* register the CTRL+C-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGINT, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGINT);
        return EXIT_FAILURE;
    }

    /* register the SIGTERM-handling function */
    stSigHandler.sa_handler = HandleStopSignals;
    iRet = sigaction(SIGTERM, &stSigHandler, NULL);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Handler registration failed for signal %d!\n",
                       SIGTERM);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
 * Catches SIGTERM and CTRL+C and cleans up before exiting
 */
void HandleStopSignals(int iSigNo)
{
    /* clean up */
    CleanUp();

    /* exit */
    exit(EXIT_SUCCESS);

    /* never reached */
    return;
}

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pCleanUp)(void))
{
    if (iRet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iRet));
        /* free resources */
        (*pCleanUp)();
        exit(EXIT_FAILURE);
    }

    return;
}

/*
 * Prints usage information
 */
void PrintUsage(const char *pcProgName)
{
    (void) printf("Usage: %s [options] <data-file>\n",
                  pcProgName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -n  --nfft <value>                   ");
    (void) printf("Number of points in FFT\n");
    (void) printf("    -p  --pfb                            ");
    (void) printf("Enable PFB\n");
    (void) printf("    -a  --nacc <value>                   ");
    (void) printf("Number of spectra to add\n");
    (void) printf("    -s  --fsamp <value>                  ");
    (void) printf("Sampling frequency\n");

    return;
}

