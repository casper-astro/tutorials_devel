/**
 * @file tut5_fileread.h
 * CASPER Tutorial 5: Heterogeneous Instrumentation
 *  Header file for file-reading functions
 *
 * @author Jayanth Chennamangalam
 * @date 2011.07.08
 */

#ifndef __TUT5_FILEREAD_H__
#define __TUT5_FILEREAD_H__

#include "tut5_main.h"

#define FILENAME_PREFIX     "file"
#define LEN_SEQ_NUM         4

/**
 * Reads all data from the input file and loads it into memory.
 */
int LoadDataToMem(void);

/**
 * Builds a formatted filename string
 */
void BuildFilename(int iCount, char acFilename[]);

/**
 * Reads one block (32MB) of data form memory.
 */
int ReadData(void);

#endif  /* __TUT5_FILEREAD_H__ */

