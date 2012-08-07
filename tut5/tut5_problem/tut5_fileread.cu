/** 
 * @file tut5_fileread.cu
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Functions to read input data files
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#include "tut5_fileread.h"

extern char4* g_pc4InBuf;
extern char4* g_pc4InBufRead;
extern char4* g_pc4DataRead_d;
extern char4* g_pc4Data_d;
extern int g_iSizeRead;
extern int g_iNTaps;
extern int g_iNFFT;
extern int g_iNumSubBands;
extern int g_iIsDataReadDone;

int g_iCurFileSeqNum = 0;
int g_iSizeFile = 0;

/* function that reads data from the data file and loads it into memory */
int LoadDataToMem()
{
    struct stat stFileStats = {0};
    int iRet = EXIT_SUCCESS;
    int iFileData = 0;
    char acFileData[LEN_GENSTRING] = {0};

    /* build the filename */
    BuildFilename(g_iCurFileSeqNum, acFileData);

    (void) printf("Opening file %s for processing...\n", acFileData);

    iRet = stat(acFileData, &stFileStats);
    if (iRet != EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Failed to stat %s: %s!\n",
                       acFileData,
                       strerror(errno));
        return EXIT_FAILURE;
    }

    g_iSizeFile = stFileStats.st_size;
    /* allocate memory if this is the first file */
    if (0 == g_iCurFileSeqNum)
    {
        g_pc4InBuf = (char4*) malloc(g_iSizeFile);
        if (NULL == g_pc4InBuf)
        {
            (void) fprintf(stderr,
                           "ERROR: Memory allocation failed! %s.\n",
                           strerror(errno));
            return EXIT_FAILURE;
        }
    }

    iFileData = open(acFileData, O_RDONLY);
    if (iFileData < EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR! Opening data file %s failed! %s.\n",
                       acFileData,
                       strerror(errno));
        return EXIT_FAILURE;
    }

    iRet = read(iFileData, g_pc4InBuf, g_iSizeFile);
    if (iRet < EXIT_SUCCESS)
    {
        (void) fprintf(stderr,
                       "ERROR: Data reading failed! %s.\n",
                       strerror(errno));
        (void) close(iFileData);
        return EXIT_FAILURE;
    }
    else if (iRet != stFileStats.st_size)
    {
        (void) printf("File read done!\n");
    }

    (void) close(iFileData);

    /* set the read pointer to the beginning of the data array */
    g_pc4InBufRead = g_pc4InBuf;

    /* increment the file sequence number */
    ++g_iCurFileSeqNum;

    return EXIT_SUCCESS;
}

/*
 * void BuildFilename(int iCount, char acFilename[])
 *
 * Builds a formatted filename string
 */
void BuildFilename(int iCount, char acFilename[])
{
    char acCount[LEN_SEQ_NUM+1] = {0};
    char acTemp[2] = {0};
    int iDigits[LEN_SEQ_NUM] = {0};
    int iNumDigits = 0;
    int i = 0;

    /* convert iCount to acCount */
    for (i = 0; i < LEN_SEQ_NUM; ++i)
    {
        if (0 == (iCount / 10))
        {
            iDigits[i] = iCount % 10;
            iNumDigits = i + 1;
            break;
        }
        else
        {
            iDigits[i] = iCount % 10;
            iCount = iCount / 10;
        }
    }
    for (i = (LEN_SEQ_NUM - iNumDigits); i > 0; --i)
    {
        (void) strcat(acCount, "0");
    }
    for (i = iNumDigits; i > 0; --i)
    {
        (void) sprintf(acTemp, "%d", iDigits[i-1]);
        (void) strcat(acCount, acTemp);
    }

    (void) sprintf(acFilename,
                   "%s%s",
                   FILENAME_PREFIX,
                   acCount);

    return;
}

/* function that reads data from input buffer */
int ReadData()
{
    /*************************************************************************/
    /* Task B: Copy new data to the device write buffer */
    CUDASafeCallWithCleanUp(cudaMemcpy(g_pc4Data_d,
                                       g_pc4InBufRead,
                                       g_iSizeRead,
                                       cudaMemcpyHostToDevice));
    /*************************************************************************/
    /* update the read pointer to where data needs to be read in from, in the
       next read */
    g_pc4InBufRead += ((g_iSizeRead
                        - ((g_iNTaps - 1)
                           * g_iNumSubBands
                           * g_iNFFT
                           * sizeof(char4)))
                       / sizeof(char4));
    /* whenever there is a read, reset the read pointer to the beginning */
    g_pc4DataRead_d = g_pc4Data_d;
    /* BUG: won't read last block */
    if ((((char *) g_pc4InBuf) + g_iSizeFile) - ((char *) g_pc4InBufRead)
        <= g_iSizeRead)
    {
        (void) printf("Data read done!\n");
        g_iIsDataReadDone = TRUE;
    }

    return EXIT_SUCCESS;
}

