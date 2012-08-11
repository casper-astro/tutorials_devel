/**
 * @file tut5_plot.h
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Header file for plotting
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#ifndef __TUT5_PLOT_H__
#define __TUT5_PLOT_H__

#include <cpgplot.h>    /* for cpg*() */
#include <math.h>       /* for log10f() in Plot() */

#include "tut5_main.h"

#define PG_DEV              "1/XS"
#define PG_VP_ML            0.10    /* left margin */
#define PG_VP_MR            0.90    /* right margin */
#define PG_VP_MB            0.12    /* bottom margin */
#define PG_VP_MT            0.98    /* top margin */
#define PG_SYMBOL           2
#define PG_CI_DEF           1
#define PG_CI_PLOT          11

int InitPlot(void);
void Plot(void);

#endif  /* __TUT5_PLOT_H__ */

