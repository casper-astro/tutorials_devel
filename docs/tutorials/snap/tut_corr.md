# Tutorial 4: Wideband Pocket Correlator

## Introduction ##
In this tutorial, you will create a simple Simulink design which uses the [iADC](https://casper.berkeley.edu/wiki/ADC2x1000-8) board on [ROACH](https://github.com/casper-astro/casper-hardware/wiki/ROACH2) and the CASPER DSP blockset to process a wideband (400MHz) signal, channelize it and output the visibilities through ROACH's PPC.

By this stage, it is expected that you have completed [tutorial 1](tut_intro.html) and [tutorial 2](tut_ten_gbe.html) and are reasonably comfortable with Simulink and basic Python. We will focus here on higher-level design concepts, and will provide you with low-level detail preimplemented.

## Background ##
Some of this design is similar to that of the previous tutorial, the Wideband Spectrometer. So completion of [tutorial 3](tut_spec.html) is recommended.

### Interferometry ###
In order to improve sensitivity and resolution, telescopes require a large collection area. Instead of using a single, large dish which is expensive to construct and complicated to maneuver, modern radio telescopes use interferometric arrays of smaller dishes (or other antennas). Interferometric arrays allow high resolution to be obtained, whilst still only requiring small individual collecting elements.

### Correlation ###
Interferometric arrays require the relative phases of antennas' signals to be measured. These can then be used to construct an image of the sky. This process is called correlation and involves multiplying signals from all possible antenna pairings in an array. For example, if we have 3 
antennas, A, B and C, we need to perform correlation across each pair, AB, AC and BC. We 
also need to do auto-correlations, which will give us the power in each signal. ie AA, BB, CC. We will 
see this implemented later. The complexity of this calculation scales with the number of antennas squared. Furthermore, it is a difficult signal routing problem since every antenna must be able to exchange data with every other antenna.

### Polarization ###
Dish type receivers are typically dual polarized (horizontal and vertical feeds). Each polarization is fed into separate ADC inputs. When correlating these antennae, we differentiate between full Stokes correlation or a half Stokes method. A full Stokes correlator does cross correlation between the different polarizations (ie for a given two antennas, A and B, it multiplies the horizontal feed from A with the vertical feed from B and vice-versa). A half stokes correlator only correlates like polarizations with each other, thereby halving the compute requirements. 

### The Correlator ###
The correlator we will be designing is a 4 input correlator as shown below. It uses 2 inputs from each of two ADCs.  It can be thought of as a 2-input full Stokes correlator or as a four input single polarization correlator.

![](../../_static/img/tut_corr/Roach_with_iadcs_on_bench.jpg)

## Setup ##
The lab at the workshop is preconfigured with the CASPER libraries, Matlab and Xilinx tools. Start Matlab.

## Creating Your Design ##

### Create a new model ###
Having started Matlab, open Simulink (either by typing simulink on the Matlab command line, or by clicking the Simulink icon in the taskbar). Create a new model and add the Xilinx System Generator and SNAP platform blocks as before in Tutorial 1.

### System Generator and Platform Blocks ###
![](../../_static/img/tut_corr/sysgen_snap_platform.png)

By now you should have used these blocks a number of times. Pull the <b>System Generator</b> block into your design from the Xilinx Blockset menu under Basic Elements. The settings can be left on default.

The <b>SNAP</b> platform block can be found under the CASPER XPS System Blockset: Platform subsystem. Set the Clock Source to adc0_clk and the rest of the configuration as the default.

Make sure you have an ADC plugged into ZDOK0 to supply the FPGA's clock!

### Sync Generator ###
![](../../_static/img/tut_corr/t4_sync_gen.png)

The Sync Generator puts out a sync pulse which is used to synchronize the blocks in the design. See the CASPER memo on sync pulse generation for a detailed explanation.

This sync generator is able to synchronize with an external trigger input. Typically we connect this to a GPS's 1pps output to allow the system to reset on a second boundary after a software arm and thus know precisely the time at which an accumulation was started. To do this you can input the 1pps signal into either ADCs' sync input.
The sync pulse allows data to be tagged with a precise timestamp. It also allows multiple ROACH boards to be synchronized, which is useful if large numbers of antenna signals are being correlated.

### ADCs ###
![](../../_static/img/tut_corr/t4_adcs_jbo.png)

Connection of the ADCs is as in tutorial 3 except for the sync outputs. Here we OR all four outputs together to produce a sync pulse sampled at one quarter the rate of the ADC's sample clock. This is the simplest way of dealing with the four sync samples supplied to the FPGA on every clock cycle, but means that our system can only be synchronized to within 3 ADC sample clocks.

Logic is also provided to generate a sync manually via a software input. This allows the design to be used even in the absence of a 1 pps signal. However, in this case, the time the sync pulse occurs depends on the latency of software issuing the sync command and the FPGA signal triggering. This introduces some uncertainty in the timestamps associated with the correlator outputs.
We will not use the 1pps in this tutorial although the design has the facility to do this hardware sync'ing. 

Set up the ADCs as follows and change the second ADC board's mask parameter to adc1...

![](../../_static/img/tut_corr/t4_adc_set.png)

Throughout this design, we use CASPER's bus_create and bus_expand blocks to simplify routing and make the design easier to follow.

![](../../_static/img/tut_corr/t4_concat_block.png)

For the purposes of simulation, it can be useful to put simulation input signals into the ADCs. These blocks are pulse generators in the case of sync inputs and any analogue source for the RF inputs (noise, CW tones etc).

![](../../_static/img/tut_corr/t4_sin_wave_set.png)
![](../../_static/img/tut_corr/t4_noise_set.png)

### Control Register ###
![](../../_static/img/tut_corr/t4_ctrl_reg_jbo.png)

This part of the Simulink design sets up a software register which can be configured in software to control the correlator. Set the yellow software register's IO direction as from processor. You can find it in the CASPER_XPS System blockset. The constant block input to this register is used only for simulation.

The output of the software register goes to three slice blocks, which will pull out the individual parameters for use with configuration. The first slice block (top) is setup as follows:

![](../../_static/img/tut_corr/t4_ctrl_slice_set.png)

The slice block can be found under the Xilinx Blockset → Control Logic. The only change with the subsequent slice blocks is the Offset of the bottom bit. They are, from top to bottom, respectively,16, 17 & 18.

After each slice block we put an edge_detect block, this outputs true if a boolean input signal is true this clock and was false last clock. Found under CASPER DSP Blockset → Misc.

Next are the delay blocks. They can be left with their default settings and can be found under Xilinx Blockset → Common.

The Goto and From bocks can be found under Simulink-> Signal Routing. Label them as in the block diagram above.

### Clip Detect and status reporting ###
To detect and report signal saturation (clipping) to software, we will create a subsystem with latching inputs.

![](../../_static/img/tut_corr/t4_status_clip_jbo.png)
![](../../_static/img/tut_corr/t4_status_report.png)

The internals of this subsystem (right) consist of delay blocks, registers and cast blocks.

The delays (inputs 2 - 9) can be keep as default. Cast blocks are required as only unsigned integers can be concatenated. Set their parameters to Unsigned, 1 bit, 0 binary points Truncated Quantization, Wrapped Overflow 
and 0 Latency.

The Registers (inputs 10 - 33) must be set up with an initial value of 0 and with enable and reset ports enabled.
The status register on the output of the clip detect is set to processor in with unsigned data type and 0 binary point with a sample period of 1.

### PFBs, FFTs and Quantisers ###
The PFB FIR, FFT and the Quantizer are the heart of this design, there is one set of each for the 4 outputs of the ADCs. However, in order to save resources associated with control logic and PFB and FFT coefficient storage, the four independent filters are combined into a single simulink block. This is configured to process four independent data streams by setting the "number of inputs" parameter on the PFB_FIR and FFT blocks to 4.

![](../../_static/img/tut_corr/t4_pfb_fft_jbo.png)

Configure the PFB_FIR_generic blocks as shown below:

![](../../_static/img/tut_corr/t4_pfb_set_jbo.png)

There is potential to overflow the first FFT stage if the input is periodic or signal levels are high as shifting inside the FFT is only performed after each butterfly stage calculation. For this reason, we recommend casting any inputs up to 18 bits with the binary point at position 17 (thus keeping the range of values -1 to 1), and then ownshifting by 1 bit to place the signal in one less than the most significant bits.

The fft_wideband_real block should be configured as follows:

![](../../_static/img/tut_corr/t4_fft_set_jbo.png)

The Quantizer Subsystem is designed as seen below. The quantizer removes the bit growth that was introduced in the PFB and FFT. We can do this because we do not need the full dynamic range.

![](../../_static/img/tut_corr/t4_quant_jbo.png)

The top level view of the Quantizer Subsystem is as seen below.

![](../../_static/img/tut_corr/t4_quant_top_lvl.png)

### LEDs ###
The following sections are more periphery to the design and will only be touched on. By now you should be comfortable putting the blocks together and be able to figure out many of the values and parameters. Also feel free to consult the reference design which sits in the tutorial 4 project directory or ask any questions of the tutorial helpers.

As a debug and monitoring output we can wire up the LEDs to certain signals. We light an LED with every sync pulse. This is a sort of heartbeat showing that the design is clocking and the FPGA is running.

We light an error LED in case any ADC overflows and another if the system is reset. The fourth LED gives a visual indication of when an accumulation is complete.

ROACH's LEDs are negative logic, so when the input to the yellow block is high, the LED is off. Since this is the opposite of what you'd normally expect, we invert the logic signals with a NOT gate. 

Since the signals might be too short to light up an LED and for us to actually see it (consider the case where a single ADC sample overflows; 1/800MHz is 1.25 nS – much too short for the human eye to see) we add a negedge delay block which delays the negative edge of a block, thereby extending the positive pulse. A length of 2^23 gives about a 10ms pulse.

![](../../_static/img/tut_corr/t4_leds_jbo.png)

### ADC RMS ###
These blocks calculate the RMS values of the ADCs' input signals. We subsample the input stream by a factor of four and do a pseudo random selection of the parallel inputs to prevent false reporting of repetitive signals. This subsampled stream is squared and accumulated for 2^16 samples.

![](../../_static/img/tut_corr/t4_adc_rms.png)

### The MAC operation ###
The multiply and accumulate is performed in the dir_x (direct-x) blocks, so named because different antenna signal pairs are multiplied directly, in parallel (as opposed to the packetized correlators' X engines which process serially). 

Two sets are used, one for the even channels and another for the odd channels. Accumulation for each antenna pair takes place in BRAM using the same simple vector accumulator used in tut3. 

![](../../_static/img/tut_corr/t4_mac_op_jbo.png)

CONTROL:

The design starts by itself when the FPGA is programmed. The only control register inputs are for resetting counters and optionally sync'ing to external signal.

Sync LED provides a “heartbeat” signal to instantly see if your design is clocked sensibly.

New accumulation LED gives a visual indication of data rates and dump times.

Accumulation counter provides simple mechanism for checking if a new spectrum output is available. (poll and compare to last value)

## Software ##
The python scripts are located in the tut4 tutorial directory. We first need to run poco_init.py to program the FPGA and configure the design. Then we can run either the auto or the cross correlations plotting scripts (plot_poco_auto.py and plot_poco_cross.py). 

poco_init.py
```python
 print('Connecting to server %s on port %i... '%(roach,katcp_port)), 
     fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, 
 timeout=10,logger=logger) 
     time.sleep(1) 
     if fpga.is_connected(): 
         print 'ok\n' 
     else: 
         print 'ERROR connecting to server %s on port %i.\n'%
 (roach,katcp_port) 
         exit_fail() 
     print '------------------------' 
     print 'Programming FPGA...', 
     if not opts.skip: 
         fpga.progdev(boffile) 
         print 'done' 
     else: 
         print 'Skipped.' 
     print 'Configuring fft_shift...', 
     fpga.write_int('fft_shift',(2**32)-1) 
     print 'done' 
     print 'Configuring accumulation period...', 
     fpga.write_int('acc_len',opts.acc_len) 
     print 'done' 
     print 'Resetting board, software triggering and resetting error 
 counters...', 
     fpga.write_int('ctrl',1<<17) #arm 
     fpga.write_int('ctrl',1<<18) #software trigger 
     fpga.write_int('ctrl',0) 
     fpga.write_int('ctrl',1<<18) #issue a second trigger 
     print 'done'
```
In previous tutorials you will probably have seen very similar code to the code above. This initiates the katcp wrapper named fpga which manages the interface between the software and the hardware. fpga.progdev programs the boffile onto the FPGA and fpga.write_int writes to a register. 

poco_adc_amplitudes.py

This script outputs in the amplitudes (or power) of each signal as well as the bits used. It updates itself ever second or so.

```bash
 ADC amplitudes
 --------------
 ADC0 input I: 0.006 (0.51 bits used)
 ADC0 input Q: 0.004 (0.19 bits used)
 ADC1 input I: 0.005 (0.45 bits used)
 ADC1 input Q: 0.004 (0.19 bits used)
 -----------------------------------
```
poco_plot_auto.py

This script grabs auto-correlations from the brams and plots them. Since there are 4 inputs, 2 for each ADC there are 4 plots. Some plots will be random if there is no noise source or tone being inputted into ADC. Ie plots 3 and 4.

![](../../_static/img/tut_corr/t4_plot_auto.png)

poco_plot_cross.py
This script grabs cross-correlations from the brams and plots them. This plotshows the cross-correlation of AB.

![](../../_static/img/tut_corr/t4_plot_cross.png)