Welcome to the 2021 Tutorials

## Introduction

This document provides an overview of the tutorial setup up at the workshop and an introduction to the environment required to run the tutorials. Before you proceed you will need to decide what hardware you are planning to work with. You have the choice of 5 platforms at this workshop, the SNAP, the SKARAB, the RFSoC 4x2, RFSoc 216 and the Red Pitaya. It is recommended that anyone new to the CASPER tools target the Red Pitaya and move onto the others when they are more familiar with the environment.

## Local Configuration Information

### Accounts and login info

There are a few options for connecting to the tutorial systems. You are welcome to run the tutorials on your own computer, after cloning the tutorials git repository, you can connect via a cable to the hardware switch, or you can connect to the tutorials servers via WiFi. Note however that this WiFi network does not have access to the internet, so you will need to download the tutorial instructions before hand.

Once on the local network, you should be given an address on the 192.168.10.X subnet. You should be able to access the compile machines. The hardware is accessable via the control machine.


### Setting Up SSH
You will also need SSH to log in to the server in the tutorial room which controls and manages the ROACH/SNAP/SKARAB/Red Pitaya boards. SSH is supported natively in Linux and MacOS. If you are using a Windows laptop, you should install an SSH client, like Putty. Log in details for this server will be provided.

### Preparing to run the tutorials
In order to get ready to build and compile designs, you need to log into the vncsession you have been allocated, and set up a directory in which to work. To do this, follow the step-by-step instructions below.

2. Start a terminal, by using <CRTL+T>. In the terminal, run:
```bash
#Create and ssh tunnel for your session number:
#Run on your local machine:
ssh server-ip -L 57XX:localhost:59XX (where XX is the port number. If it were port :5, you would do 5705:localhost:5905 etc)

#Run vncviewer on your local machine, connect to: localhost:57XX

# Make a directory unique to you, when all your work will go
mkdir <directory_name> # eg. mkdir julio-iglesias
# Go into this directory
cd <directory name> #eg. cd julio-iglesias
# Grab a copy of the tutorials repository from github
git clone https://github.com/casper-astro/tutorials_devel.git
git submodule update --init <path to mlib_devel submodule>" 
```

3. Finally, start MATLAB.
```bash
#Run the following from the terminal to start MATLAB
./startsg startsg.local
```
You should be greeted with a MATLAB window. After a few seconds, it should be ready to accept user input.

4. From here, you can either open one of the tutorial .slx files using the "Open" button, or start a new simulink design by typing "simulink" in the MATLAB prompt and starting a new model.
5. Now go to the [Tutorials](https://casper-tutorials.readthedocs.io/en/latest/) page to find the instructions for your chosen tutorial. When you have compiled your design, come back here to see how to load it on to hardware.

## Getting your designs on to hardware
When your compile has finished, it should deliver you a .fpg file (this will be created in <build_dir>/bit_files/ for ROACHs, or <build_dir>/outputs/ for SNAP / SKARAB / Red Pitaya). This file needs copying to the server in the tutorial room which is connected to all the FPGA boards. To copy the file between machines, run (in a terminal within your vnc session):
```bash
scp /path/to/fpg_file.fpg casper@mex2:<name you want your file to have>.fpg
```
For example:
```bash
scp ~/julio-iglesias/tutorials_devel/vivado/snap/tut_intro/snap_tut_intro/outputs/snap_tut_intro_2017-08-13_1508.fpg hpw_hw@dbelab02:julio-iglesias_snap_intro.fpg


