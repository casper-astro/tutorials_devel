# Casper Guiyang Workshop 2018 Tutorials Help Page

## Local Configuration Information

### Accounts and login info
When you first arrive please connect to the local network in the tutorial room. If you are unable to join the wired network, you should connect to the local wireless network. The **SSID is casper2018tutorials**. **The WiFi password will be given to you when you arrive**.

Once on the local network, you should be given an address on the 10.0.1.X subnet. You should be able to access the internet, as well as all the hardware in the tutorial room.


**When you arrive at the tutorial sessions you will be allocated a remote server to build designs on, and provided with login details.**

### Setting Up SSH
You will also need SSH to log in to the server in the tutorial room which controls the ROACH/SNAP/SKARAB boards. SSH is supported natively in Linux and MacOS. If you are using a Windows laptop, you should install an SSH client, like Putty. Log in details for this server will be provided when you arrive at the tutorial sessions.

### Preparing to run the tutorials
In order to get ready to build and compile designs, you need to log into the server you have been allocated, and set up a directory in which to work. To do this, follow the step-by-step instructions below.

1. Using your ssh client, and the server details you were given at the tutorial session, connect to a remote desktop session 2. Start a terminal on this desktop, by clicking (look in the top left of the screen) Applications >> System Tools >> Terminal 
3. Copy the tutorials code repository, and associated CASPER libraries. In the terminal, run:
```bash
# Go into the home directory (you're probably already there, but let's make sure)
cd ~
# Make a directory unique to you, when all your work will go
mkdir <directory_name> # eg. mkdir julio-iglesias
# Go into this directory
cd <directory name> #eg. cd julio-iglesias
# Grab a copy of the tutorials repository from the '/home/fast' directory
cp -r /home/fast/tutorials_devel ~
```

That wasn't so hard, right? Running all the above commands might be a little bit cumbersome, but it ensures your versions of the CASPER libraries are in sync with the tutorial models you are running. This will prevent all kinds of pain later on.

2. Decide which hardware platform you are compiling for and go to the appropriate direcctory. Different directories contain slightly different MATLAB / Xilinx configurations, so it's important to choose the right one for the platform you are using.
```bash
cd ise # For ROACH only
# or...
cd vivado # For SNAP and SKARAB only
# then one of the following three commands
cd roach2 # For ROACH2 designs
cd snap   # For SNAP designs
cd skarab # For skarab designs
```

3. Finally, start MATLAB. The following command will configure your environment with the install locations of the MATLAB / Xilinx tools. This configuration depends on how you have set up your compile machines. The startsg.local files below will only work on the build machines.

```bash
./startsg
```
You should be greeted with a MATLAB window. After a few seconds, it should be ready to accept user input.

4. From here, you can either open one of the tutorial .slx files using the "Open" button, or start a new simulink design by typing "simulink" in the MATLAB prompt and starting a new model.
5. Now go to the [Tutorials](https://casper-toolflow.readthedocs.io/projects/tutorials/en/workshop2018/) page to find the instructions for your chosen tutorial. When you have compiled your design, come back here to see how to load it on to hardware.

## Getting your designs on to hardware
When your compile has finished, it should deliver you a .fpg file (this will be created in <build_dir>/bit_files/ for ROACHs, or <build_dir>/outputs/ for SNAP / SKARAB). 

You can now follow the tutorial instructions to program this file to an FPGA board. Details of the available FPGA platforms is below.

### Available FPGA hardware
There are two SNAPs, two SKARABs, and 2 ROACH2s in the tutorial room. You can access them from maze (or your laptops, if you have the casperfpga libraries installed) using the hostnames

* roach1
* roach2
* snap1
* snap2
* skarab020406-01 (40GbE port)
* skarab020802-01 (40GbE port)
