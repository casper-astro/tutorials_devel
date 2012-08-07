#!/usr/bin/python2.7

# tut5_gencoeff.py
# CASPER Tutorial 5: Heterogeneous Instrumentation
#   Generate PFB filter coefficients. The filter coefficients array will
#   contain duplicates for optimised reading from the GPU.
#
# Created by Jayanth Chennamangalam based on code by Sean McHugh, UCSB

import sys
import getopt
import math
import numpy
import matplotlib.pyplot as plotter

# function definitions
def PrintUsage(ProgName):
    "Prints usage information."
    print "Usage: " + ProgName + " [options]"
    print "    -h  --help                 Display this usage information"
    print "    -n  --nfft <value>         Number of points in FFT"
    print "    -t  --taps <value>         Number of taps in PFB"
    print "    -b  --sub-bands <value>    Number of sub-bands in data"
    print "    -d  --data-type <value>    Data type - \"float\" or "          \
          + "\"signedchar\""
    print "    -p  --no-plot              Do not plot coefficients"
    return

# default values
NFFT = 32768                # number of points in FFT
NTaps = 8                   # number of taps in PFB
NSubBands = 1               # number of sub-bands in data
DataType = "signedchar"     # data type - "float" or "signedchar"
Plot = True                 # plot flag

# get the command line arguments
ProgName = sys.argv[0]
OptsShort = "hn:t:b:d:p"
OptsLong = ["help", "nfft=", "taps=", "sub-bands=", "data-type=", "no-plot"]

# check if the minimum expected number of arguments has been passed
# to the program
if (1 == len(sys.argv)):
    sys.stderr.write("ERROR: No arguments passed to the program!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# get the arguments using the getopt module
try:
    (Opts, Args) = getopt.getopt(sys.argv[1:], OptsShort, OptsLong)
except getopt.GetoptError, ErrMsg:
    # print usage information and exit
    sys.stderr.write("ERROR: " + str(ErrMsg) + "!\n")
    PrintUsage(ProgName)
    sys.exit(1)

# parse the arguments
for o, a in Opts:
    if o in ("-h", "--help"):
        PrintUsage(ProgName)
        sys.exit()
    elif o in ("-n", "--nfft"):
        NFFT = int(a)
    elif o in ("-t", "--taps"):
        NTaps = int(a)
    elif o in ("-b", "--sub-bands"):
        NSubBands = int(a)
    elif o in ("-d", "--data-type"):
        DataType = a
    elif o in ("-p", "--no-plot"):
        Plot = False
    else:
        PrintUsage(ProgName)
        sys.exit(1)

M = NTaps * NFFT

# the filter-coefficient-generation section -->
X = numpy.array([(float(i) / NFFT) - (float(NTaps) / 2) for i in range(M)])
PFBCoeff = numpy.sinc(X) * numpy.hanning(M)
# <-- the filter-coefficient-generation section

# create conversion map
if ("signedchar" == DataType):
    Map = numpy.zeros(256, numpy.float32)
    for i in range(0, 128):
        Map[i] = float(i) / 128
    for i in range(128, 256):
        Map[i] = - (float(256 -i) / 128)

# 32-bit (float) coefficients
PFBCoeffFloat32 = numpy.zeros(M * NSubBands, numpy.float32)
# 8-bit (signedchar) coefficients
if ("signedchar" == DataType):
    PFBCoeffInt8 = numpy.zeros(M * NSubBands, numpy.int8)
k = 0
for i in range(len(PFBCoeff)):
    Coeff = float(PFBCoeff[i])
    if ("signedchar" == DataType):
        for j in range(256):
            #if (math.fabs(Coeff - Map[j]) <= (0.0078125 / 2)):
            if (math.fabs(Coeff - Map[j]) <= 0.0078125):
                for m in range(NSubBands):
                    PFBCoeffInt8[k+m] = j
                break
    elif ("float" == DataType):
        for m in range(NSubBands):
            PFBCoeffFloat32[k+m] = Coeff
    else:
        # print usage information and exit
        sys.stderr.write("ERROR: Invalid data type!\n")
        PrintUsage(ProgName)
        sys.exit(1)
    k = k + NSubBands

# write the coefficients to disk and also plot it
FileCoeff = open("coeff_"                                                     \
                  + DataType + "_"                                            \
                  + str(NTaps) + "_"                                          \
                  + str(NFFT) + "_"                                           \
                  + str(NSubBands) + ".dat",                                  \
                 "wb")
if ("signedchar" == DataType):
    FileCoeff.write(PFBCoeffInt8)
    # plot the coefficients
    if (Plot):
        plotter.plot(PFBCoeffInt8)
else:
    FileCoeff.write(PFBCoeffFloat32)
    # plot the coefficients
    if (Plot):
        plotter.plot(PFBCoeffFloat32)

FileCoeff.close()

if (Plot):
    plotter.show()

