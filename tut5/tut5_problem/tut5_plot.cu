/** 
 * @file tut5_plot.cu
 * CASPER Workshop 2011 Tutorial 5: Heterogeneous Instrumentation
 *  Plotting-related functions
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#include "tut5_plot.h"

extern int g_iNumSubBands;
extern int g_iNFFT;
extern float4* g_pf4SumStokes;

float* g_pfSumPowX = NULL;
float* g_pfSumPowY = NULL;
float* g_pfSumStokesRe = NULL;
float* g_pfSumStokesIm = NULL;
float* g_pfFreq = NULL;
float g_fFSamp = 1.0;                   /* 1 [frequency] */

int InitPlot()
{
    int iRet = EXIT_SUCCESS;
    int i = 0;

    iRet = cpgopen(PG_DEV);
    if (iRet <= 0)
    {
        (void) fprintf(stderr,
                       "ERROR: Opening graphics device %s failed!\n",
                       PG_DEV);
        return EXIT_FAILURE;
    }

    cpgsch(3);
    cpgsubp(g_iNumSubBands, 4);

    g_pfSumPowX = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumPowX)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumPowY = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumPowY)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesRe = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumStokesRe)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfSumStokesIm = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfSumStokesIm)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }
    g_pfFreq = (float*) malloc(g_iNFFT * sizeof(float));
    if (NULL == g_pfFreq)
    {
        (void) fprintf(stderr,
                       "ERROR: Memory allocation failed! %s.\n",
                       strerror(errno));
        return EXIT_FAILURE;
    }

    /* load the frequency axis */
    for (i = 0; i < g_iNFFT; ++i)
    {
        g_pfFreq[i] = ((float) i * g_fFSamp) / g_iNFFT;
    }

    return EXIT_SUCCESS;
}

void Plot()
{
    float fMinFreq = g_pfFreq[0];
    float fMaxFreq = g_pfFreq[g_iNFFT-1];
    float fMinY = FLT_MAX;
    float fMaxY = -(FLT_MAX);
    int i = 0;
    int j = 0;
    int k = 0;

    for (k = 0; k < g_iNumSubBands; ++k)
    {
        for (i = k, j = 0;
             i < (g_iNumSubBands * g_iNFFT);
             i += g_iNumSubBands, ++j)
        {
            if (0.0 == g_pf4SumStokes[i].x)
            {
                g_pfSumPowX[j] = 0.0;
            }
            else
            {
                g_pfSumPowX[j] = g_pf4SumStokes[i].x;
            }
            if (0.0 == g_pf4SumStokes[i].y)
            {
                g_pfSumPowY[j] = 0.0;
            }
            else
            {
                g_pfSumPowY[j] = g_pf4SumStokes[i].y;
            }
            g_pfSumStokesRe[j] = g_pf4SumStokes[i].z;
            g_pfSumStokesIm[j] = g_pf4SumStokes[i].w;
        }

        /* plot accumulated X-pol. power */
        fMinY = FLT_MAX;
        fMaxY = -(FLT_MAX);
        for (i = 0; i < g_iNFFT; ++i)
        {
            if (g_pfSumPowX[i] > fMaxY)
            {
                fMaxY = g_pfSumPowX[i];
            }
            if (g_pfSumPowX[i] < fMinY)
            {
                fMinY = g_pfSumPowX[i];
            }
        }
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        for (i = 0; i < g_iNFFT; ++i)
        {
            g_pfSumPowX[i] -= fMaxY;
        }
        fMinY -= fMaxY;
        fMaxY = 0;
        cpgpanl(k + 1, 1);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumPowX");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumPowX);
        cpgsci(PG_CI_DEF);

        /* plot accumulated Y-pol. power */
        fMinY = FLT_MAX;
        fMaxY = -(FLT_MAX);
        for (i = 0; i < g_iNFFT; ++i)
        {
            if (g_pfSumPowY[i] > fMaxY)
            {
                fMaxY = g_pfSumPowY[i];
            }
            if (g_pfSumPowY[i] < fMinY)
            {
                fMinY = g_pfSumPowY[i];
            }
        }
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        for (i = 0; i < g_iNFFT; ++i)
        {
            g_pfSumPowY[i] -= fMaxY;
        }
        fMinY -= fMaxY;
        fMaxY = 0;
        cpgpanl(k + 1, 2);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumPowY");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumPowY);
        cpgsci(PG_CI_DEF);

        /* plot accumulated real(XY*) */
        fMinY = FLT_MAX;
        fMaxY = -(FLT_MAX);
        for (i = 0; i < g_iNFFT; ++i)
        {
            if (g_pfSumStokesRe[i] > fMaxY)
            {
                fMaxY = g_pfSumStokesRe[i];
            }
            if (g_pfSumStokesRe[i] < fMinY)
            {
                fMinY = g_pfSumStokesRe[i];
            }
        }
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        cpgpanl(k + 1, 3);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumStokesRe");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesRe);
        cpgsci(PG_CI_DEF);

        /* plot accumulated imag(XY*) */
        fMinY = FLT_MAX;
        fMaxY = -(FLT_MAX);
        for (i = 0; i < g_iNFFT; ++i)
        {
            if (g_pfSumStokesIm[i] > fMaxY)
            {
                fMaxY = g_pfSumStokesIm[i];
            }
            if (g_pfSumStokesIm[i] < fMinY)
            {
                fMinY = g_pfSumStokesIm[i];
            }
        }
        /* to avoid min == max */
        fMaxY += 1.0;
        fMinY -= 1.0;
        cpgpanl(k + 1, 4);
        cpgeras();
        cpgsvp(PG_VP_ML, PG_VP_MR, PG_VP_MB, PG_VP_MT);
        cpgswin(fMinFreq, fMaxFreq, fMinY, fMaxY);
        //cpglab("Bin Number", "", "SumStokesIm");
        cpgbox("BCNST", 0.0, 0, "BCNST", 0.0, 0);
        cpgsci(PG_CI_PLOT);
        cpgline(g_iNFFT, g_pfFreq, g_pfSumStokesIm);
        cpgsci(PG_CI_DEF);
    }

    return;
}

