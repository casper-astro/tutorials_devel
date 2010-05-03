The tutorials at the 2010 CASPER workshop are as follows:

0) Pre Tutorial VNC Setup
This year the workshops will be run through VNC servers, this document show how to configure the VNC client on windows and linux systems to access the workshop servers.

1) Introduction <COMPLETE>
Introduction to Simulink, communication with FPGAs. Build a simple program that flashes LEDs, counts and adds numbers on demand, captures raw ADC data. Communicate with ROACH's FPGA using the PPC's BORPH operating system.  Prerequisites: just your enthusiastic self!

2) 10GbE, KATCP <10GbE BROKEN - recompile with XILINX PHY> <DOC OUTSTANDING>
Construct a design that transmits a counter over 10GbE to another ROACH port. During this tutorial, you will learn more about the CASPER hardware interfaces, communicating with ROACH remotely using KATCP and the supplied Python libraries. Prerequisites: Introduction tutorial (we expect conceptual understanding of software registers and PPC/FPGA mapped memory regions along with some basic Simulink experience). Also useful, but not required, is some experience in programming in Python.

3) Wideband spectrometer <DESIGN FAILS TIMING> <SOFTWARE OUTSTANDING>
Build a 512MHz "wideband" spectrometer with 2048 FFT channels on ROACH. This will introduce the ADC boards, the CASPER DSP libraries including the wideband PFB and FFT blocks as well as demonstrate the use of vector accumulators, shared BRAMs and readout through the PPC's 1GbE port. We will make use of KATCP for remote control of the ROACHes using Python libraries to plot the output. Prerequisites: tutorials 1 and 2.

4) Narrowband pocket correlator <DESIGN FROM ALAN, Else WIDEBAND version is working and complete with control software>
Build a 250MHz, 1024 channel single board correlator on ROACH. This tutorial demonstrates the use of CASPER's digital downmix and filter blocks, along with the narrowband biplexed PFB and FFT blocks. Data is accumulated on the FPGA and output through the PPC's 1GbE port and displayed remotely using KATCP. Prerequisites: tutorials 1 and 2.

5) High resolution spectrometer <DESIGN AND DOC FROM DAVE>
Build a 500MHz, 1 million channel spectrometer on a single ROACH board. This tutorial demonstrates advanced use of the PFB and FFT blocks, some DSP concepts, along with the ROACH's DRAM for large-scale accumulation. Prerequisites: tutorials 1 and 2 plus either 3 or 4.


Each tutorial folder includes a pdf walkthrough of the design, the Simulink design model file, a pre-compiled bof file for loading onto a ROACH board and any required software.

In addition, David George will give a presentation on constructing "yellow blocks", the hardware interfaces. This will be in the form of an interface to the original CASPER iADC. Included is an overview of hardware blocks in Simulink and an overview of OPB/EPB busses.
