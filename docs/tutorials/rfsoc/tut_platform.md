# Tutorial 1: RFSoC First Design

In this tutorial, you will make a simple design for an rfsoc board using the CASPER toolflow.
It will take you through launching the toolflow, creating a valid CASPER design in Simulink,
generating an fpg file, programming the fpg file to a CASPER rfsoc board, and interacting with
the running hardware using the casperfpga library through a python interface.

This tutorial assumes that you have already setup your environment correctly, as explained in the
Getting Started With RFSoC tutorial. You should have all the programs and packages installed and 
configuration files set, and you should have successfully set up and tested your connection to the rfsoc board.

## Creating Your First Design
### Create a New Model
Make sure you are in your previously set up environment and navigate to mlib_devel.
Start Matlab by exectuing `startsg`. This will properly load the Xilinx and CASPER libraries into Simulink, so
long as your `config.local` file is set correctly. Within Matlab, start Simulink by typing `simulink` into
Matlab's command line. Create a new blank model and save it with an appropriate name.

### Library Organization

There are three primary libraries in Simulink you will use when designing for your rfsoc board:

1. The **CASPER XPS Library** contains the CASPER "Yellow Blocks". These blocks encapsulate interfaces to
your board's hardware (ADCs, memory chips, CPUs, various ports, etc).

2. The **CASPER DSP Library** contains (often green) blocks that implement DSP functions (filters, FFTs, etc).

3. The **Xilinx Library** contains blue blocks which provide low-level fpga functionality (multiplexing,
delaying, adding, etc). It also contains the *System Generator* block, which contains information about the FPGA
you are using.

### Add the Xilinx System Generator and XSG core config blocks
The first thing to add is the *System Generator* block, found using the Simulink Library Browser 
in Xilinx Blockset->Basic Elements->System Generator. Add the block by clicking and dragging the 
block into your design. See the Simulink documentation by mathworks for other methods of finding and adding 
blocks to your design.

You can double click on the added block to see its configuration. However, instead of configuring the System
Generator ourselves, we will use a platform block from the **CASPER XPS Library** to configure it. Locate the block for the
board you are using in CASPER XPS Blockset->Platforms-><your platform>. This example uses the ZCU216 board, so
this example will add the ZCU216 block.

Double click on the added platform block to see its configuration. Confirm that the Hardware Platform parameter matches
the platform you are using. From here, you can also configure other options, such as the board clocks.

This yellow platform block will automatically configure the System Generator block for you, and we can now move on to
the design.

Note: **The System Generator and XPS platform blocks are required by all CASPER designs**

### The Example Design
In order to demonstrate the basic use of hardware interfaces and software interaction, this design will implement 3 
different functions on the board:

1. A Flashing LED
2. A Software Controllable Counter
3. A Software Controllable Adder

### Function 1: Flashing LED
We can create a flashing LED by using a 27 bit counter. On the ZCU216, the default clock given by its CASPER platform
block is 250 MHz, which will toggle the most significant bit on the 27 bit counter about every 0.27 seconds. The 
principle is the same for any clock rate on any board. We can output this most significant bit to an LED on the board,
causing the LED to flash at about 50% duty cycle every so many seconds (half a second for this example).

#### Step 1: Add a counter
Add a blue counter block to the design. It can be found in Xilinx Blockset->Basic Elements->Counter.

Double click the block to access its parameters, and set it to free running, 27 bits, unsigned. This will set the
counter to count from 0 to (2^27)-1, wrap back to zero, and continue.

#### Step 2:



