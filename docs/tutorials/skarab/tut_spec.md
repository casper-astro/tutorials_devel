# Tutorial 4: Wideband Spectrometer

## Introduction ##
A spectrometer is something that takes a signal in the time domain and converts it to the frequency domain. In digital systems, this is generally achieved by utilising the FFT (Fast Fourier Transform) algorithm.

When designing a spectrometer for astronomical applications, it's important to consider the 	science case behind it. For example, pulsar timing searches will need a spectrometer which can 	dump spectra on short timescales, so the rate of change of the spectra can be observed. In contrast, a deep field HI survey will accumulate multiple spectra to increase the signal to noise ratio. It's also important to note that “bigger isn't always better”; the higher your spectral and time resolution are, the more data your computer (and scientist on the other end) will have to deal with. For now, let's skip the science case and familiarize ourselves with an example spectrometer.

## Setup ##

This tutorial comes with a completed model file, a compiled bitstream, ready for execution on Skarab, as well as a Python script to configure the Skarab and make plots. [Here](https://github.com/casper-astro/tutorials_devel/tree/master/skarab/tut_spec)

## Spectrometer Basics ##

When designing a spectrometer there are a few main parameters of note:
	

- **Bandwidth**: The width of your frequency spectrum, in Hz. This depends on the sampling rate; for complex sampled data this is equivalent to:

![](../../_static/img/skarab/tut_spec/bandwidtheq1.png)

In contrast, for real or Nyquist sampled data the rate is half this:

![](../../_static/img/skarab/tut_spec/bandwidtheq2.png)

as two samples are required to reconstruct a given waveform .

- **Frequency resolution**: The frequency resolution of a spectrometer, Δf, is given by

![](../../_static/img/skarab/tut_spec/freq_eq.png),

and is the width of each frequency bin. Correspondingly, Δf is a measure of how precise you can measure a frequency.

- **Time resolution**: Time resolution is simply the spectral dump rate of your instrument. We generally accumulate multiple spectra to average out noise; the more accumulations we do, the lower the time resolution. For looking at short timescale events, such as pulsar bursts, higher time resolution is necessary; conversely, if we want to look at a weak HI signal, a long accumulation time is required, so time resolution is less important.

## Simulink / CASPER Toolflow ##

### Simulink Design Overview ###
If you're reading this, then you've already managed to find all the tutorial files.  By now, I presume you can open the model file and have a vague idea of what's happening.
The best way to understand fully is to follow the arrows, go through what each block is doing and make sure you know why each step is done. To help you through, there's some “blockumentation” in the appendix, which should (hopefully) answer all questions you may have. A brief rundown before you get down and dirty:

- In the slx file, you'll notice a subsystem block in the top left corner of the design.  If you click into it, you'll see it contains a 40 GbE core.  That core used to instantiante the board support package and the microblaze controller, but it is no longer the case. The board support package and the microblaze controller are now part of the Skarab platform block. However, the 40 GbE or 1 GbE core is still required in a Skarab design for communication purposes.

- The all important Xilinx token is placed to allow System Generator to be called to compile the design.

- In the MSSGE block, the hardware type is set to “SKARAB:xc7vx690t” and clock rate is specified as 187.5MHz.  This frequency is specially chosen to avoid overflows on the ADC.  Implementing other clock frequencies will require you to use the data valid port leaving the ADC yellow block.

- The input signal is digitised by the ADC, resulting in eight parallel time samples of 16 bits each clock cycle: four in i and four in q. The ADC runs at 3 GHz but decimates by a factor of four, which gives a 375 MHz nyquist sampled spectrum.  The Skarab ADC uses a digital downconverter, which has a default frequency of 1 GHz. The output range is a signed number in the range -1 to +1 (ie 15 bits after the decimal point). This is expressed as fix_16_15.

- Unlike the other CASPER spectrometer tutorials, we use the complex FFT block here.  The Skarab ADC produces demultiplexed i and q channels that are concatenated and fed to the FFT block.

- You may notice Xilinx delay blocks dotted all over the design. It's common practice to add these into the design as it makes it easier to fit the design into the logic of the FPGA. It consumes more resources, but eases signal timing-induced placement restrictions.

- The real and imaginary (sine and cosine value) components of the FFT are plugged into power blocks, to convert from complex values to real power values by squaring.

- The requantized signals then enter the vector accumulators, simple_bram_vacc0 through simple_bram_vacc3, which are 64 bit vector accumulators. Accumulation length is controlled by the acc_cntrl block.

- The accumulated signal is then fed into software registers, mem1 through mem4.


Without further ado, open up the model file and start clicking on things, referring to the blockumentation as you go.

### [adc](https://casper.berkeley.edu/wiki/Adc) ###

![](../../_static/img/skarab/tut_spec/skarab_ADC.png)

The first step to creating a frequency spectrum is to digitize the signal. This is done with an ADC – an Analogue to Digital Converter.

The ADC block converts analog inputs to digital outputs. Every clock cycle, the inputs are sampled and digitized to 16 bit binary point numbers in the range of -1 to 1 and are then output by the ADC. This is achieved through the use of two's-compliment representation with the binary point placed after the seven least significant bits. This means we can represent numbers from -32768 through to 32767 including the number 0. Simulink represents such numbers with a fix_16_15 moniker.

ADCs often internally bias themselves to halfway between 0 and -1. This means that you'd typically see the output of an ADC toggling between zero and -1 when there's no input. It also means that unless otherwise calibrated, an ADC will have a negative DC offset.

The Skarab ADC is clocked at 3.0 GHz.  There is a decimation factor of 4, so the sample rate is 750 MHz.  The i and q channels each have a demux factor of 4, so the FPGA is clocked at 187.5 MHz. The bandwidth for a 750 MHz sample rate is 375 MHz, as Nyquist sampling requires two samples (or more) each second.

**INPUTS**

|                   Port                  | Description                                                                                                               |
| --------------------------------------- |---------------------------------------------------------------------------------------------------------------------------|
| adc_sync_start_in | The ADC takes a sync pulse, which we connect to a reset register. |
| pll_sync_start_in                                | The ADC has a port for a reference clock but we do not use one here.                                   |

**OUTPUTS**

The Skarab ADC has four channels and a series of eight outputs for each.  The outputs for a channel comprise four demultiplexed i's and four demultiplexed q's.  The i0 port is concatenated with the q0 port to form a complex stream using the real/imaginary-to-complex block.  Similarly, the i1 port is concatenated with the q1 port, i2 with q2, and i3 with q3.


### [fft](https://casper.berkeley.edu/wiki/fft) ###

![](../../_static/img/skarab/tut_spec/wideband_fft.png)

The FFT block is the most important part of the design to understand. The cool green of the FFT block hides the complex and confusing FFT butterfly biplex algorithms that are under the hood. You do need to have a working knowledge of it though, so I recommend reading Chapter 8 and Chapter 12 of Smith's free online DSP guide at (http://www.dspguide.com/). Parts of the documentation below are taken from the [[Block_Documentation | block documentation]] by Aaron Parsons and Andrew Martens.

**INPUTS/OUTPUTS**

| Port   | Description |
| --- | --- |
| sync  | Like many of the blocks, the FFT needs a heartbeat to keep it sync'd. |
| shift  | Sets the shifting schedule through the FFT. Bit 0 specifies the behavior of stage 0, bit 1 of stage 1, and so on. If a stage is set to shift (with bit = 1), then every sample is divided by 2 at the output of that stage. We've set Shift to 2^(13 − 1) − 1, which will shift the data by 1 on every stage to prevent overflows. |
| in0-3  | Complex-valued inputs. |
| out0-4 | This real FFT produces four simultaneous outputs. Each of these parallel FFT outputs will produce sequential channels of complex samples on every clock cycle. So, on the first clock cycle (after a sync pulse, which denotes the start), you'll get frequency channel zero and frequency channel one. Each of those are complex numbers. Then, on the second clock cycle, you'll get frequency channels 2 and 3. These are followed by 4 and 5 etc etc. So we chose to label these output paths "even" and "odd", to differentiate the path outputting channels 0,2,4,6,8...N-1 from the channel doing 1,3,5,7...N. As you can see, in order to recreate the full spectrum, we need to interleave these paths to produce 0,1,2,3,4,5...N. Following the lines you'll see that these two inputs end up in an “odd” and “even” shared BRAMs. These are then interleaved in the tut_spec.py script to form a complete spectrum. |

**PARAMETERS**

| Parameter | Description |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Size of FFT | How many points the FFT will have. We've selected 2^12 = 4096 points. |
| Input/output bitwidth | The number of bits in each real and imaginary sample as they are carried through the FFT. Each FFT stage will round numbers back down to this number of bits after performing a butterfly computation. This has to match what the pfb_fir is throwing out. The default is 18 so this shouldn't need to be changed. |
| Coefficient bitwidth | The amount of bits for each coefficient. 18 is default. |
| Number of simultaneous inputs | The number of parallel time samples which are presented to the FFT core each clock. We have 2^2 = 4 parallel data streams, so this should be set to 2. |
| Unscramble output | Some reordering is required to make sure the frequency channels are output in canonical frequency order. If you're absolutely desperate to save as much RAM and logic as possible you can disable this processing, but you'll have to make sure you account for the scrambling of the channels in your downstream software. For now, because our design will comfortably fit on the FPGA, leave the unscramble option checked. |
| Overflow Behavior | Indicates the behavior of the FFT core when the value of a sample exceeds what can be expressed in the specified bit width. Here we're going to use Wrap, since Saturate will not make overflow corruption better behaved. |
| Add Latency | Latency through adders in the FFT. Set this to 2. |
| Mult Latency | Latency through multipliers in the FFT. Set this to 3.  |
| BRAM Latency | Latency through BRAM in the FFT. Set this to 2. |
| Convert Latency | Latency through blocks used to reduce bit widths after twiddle and butterfly stages. Set this to 1. |
| Input Latency | Here you can register your input data streams in case you run into timing issues. Leave this set to 0. |
| Latency between biplexes and fft_direct | Here you can add optional register stages between the two major processing blocks in the FFT. These can help a failing design meet timing. For this tutorial, you should be able to compile the design with this parameter set to 0. |
| Architecture | Set to Virtex5, the architecture of the FPGA on the Skarab. This changes some of the internal logic to better optimise for the DSP slices. If you were using an older iBOB board, you would need to set this to Virtex2Pro. |
| Use less | This affects the implementation of complex multiplication in the FFT, so that they either use fewer multipliers or less logic/adders. For the complex multipliers in the FFT, you can use 4 multipliers and 2 adders, or 3 multipliers and a bunch or adders. So you can trade-off DSP slices for logic or vice-versa. Set this to Multipliers. |
| Number of bits above which to store stage's coefficients in BRAM | Determines the threshold at which the twiddle coefficients in a stage are stored in BRAM. Below this threshold distributed RAM is used. By changing this, you can bias your design to use more BRAM or more logic. We're going to set this to 8. |
| Number of bits above which to store stage's delays in BRAM | Determines the threshold at which the twiddle coefficients in a stage are stored in BRAM. Below this threshold distributed RAM is used. Set this to 9. |
| Multiplier Implementation | Determines how multipliers are implemented in the twiddle function at each stage. Using behavioral HDL allows adders following the multiplier to be folded into the DSP48Es in Virtex5 architectures. Other options choose multiplier cores which allows quicker compile time. You can enter an array of values allowing exact specification of how multipliers are implemented at each stage. Set this to 1, to use embedded multipliers for all FFT stages. |
| Hardcode shift schedule | If you wish to save logic, at the expense of being able to dynamically specify your shifting regime using the block's "shift" input, you can check this box. Leave it unchecked for this tutorial. |
| Use DSP48's for adders | The butterfly operation at each stage consists of two adders and two subtracters that can be implemented using DSP48 units instead of logic. Leave this unchecked. |

### [power](https://casper.berkeley.edu/wiki/Power) ###

![](../../_static/img/skarab/tut_spec/skarab_power.png)

The power block computes the power of a complex number. The power block typically has a latency of 5 and will compute the power of its input by taking the sum of the squares of its real and imaginary components.  The power block is written by Aaron Parsons and online documentation is by Ben Blackman.

In our design, there are two power blocks, which compute the power of the odd and even outputs of the FFT. The output of the block is 36.34 bits; the next stage of the design re-quantizes this down to a lower bitrate.

**INPUTS/OUTPUTS**

| Port | Direction | Data Type | Description |
|-------|-----------|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| c | IN | 2*BitWidth Fixed point | A complex number whose higher BitWidth bits are its real part and lower BitWidth bits are its imaginary part. |
| power | OUT | UFix_(2*BitWidth)_(2*BitWidth-1) | The computed power of the input complex number. |

**PARAMETERS**

| Parameter | Variable | Description |
|-----------|----------|----------------------------------|
| Bit Width | BitWidth | The number of bits in its input. |



### simple_bram_vacc ###

![](../../_static/img/skarab/tut_spec/memory.png)

The simple_bram_vacc block is used in this design for vector accumulation. Vector growth is approximately 28 bits each second, so if you wanted a really long accumulation (say a few hours), you'd have to use a block such as the qdr_vacc or dram_vacc. As the name suggests, the simple_bram_vacc is simpler so it is fine for this demo spectrometer.
The FFT block demultiplexed frequency bins directly to the accumulator and memory blocks.  These streams are multiplexed in softawre using the tut_spec.py script.

**PARAMETERS**

| Parameter | Description |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Vector length | The length of the input/output vector. The FFT block produces four streams of 2048 length, so we set this to 2048. |
| no. output bits | As there is bit growth due to accumulation, we need to set this higher than the input bits. The input is 36.35 from the FFT block, so we have set this to 64 bits. |
| Binary point (output) | Since we are accumulating 36.35 values there should be 35 bits below the binary point of the output, so set this to 35. |

**INPUTS/OUTPUTS**

| Port | Description |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| new_acc | A boolean pulse should be sent to this port to signal a new accumulation. We can't directly use the sync pulse, otherwise this would reset after each spectrum. So, Jason has connected this to acc_cntrl, a block which allows us to set the accumulation period. |
| din/dout | Data input and output. The output depends on the no. output bits parameter. |
| Valid | The output of this block will only be valid when it has finished accumulating (signalled by a boolean pulse sent to new_acc). This will output a boolean 1 while the vector is being output, and 0 otherwise. |



### [Software Registers](https://casper.berkeley.edu/wiki/Software_register) ###

There are a few [control registers](https://casper.berkeley.edu/wiki/Software_register), led blinkers, and [snap](https://casper.berkeley.edu/wiki/Snap) block dotted around the design too:

- **cnt_rst**: Counter reset control. Pulse this high to reset all counters back to zero.

- **acc_len**: Sets the accumulation length. Have a look in tut_spec.py for usage.

- **sync_cnt**: Sync pulse counter. Counts the number of sync pulses issued. Can be used to figure out board uptime and confirm that your design is being clocked correctly.

- **acc_cnt**: Accumulation counter. Keeps track of how many accumulations have been done.

- **led0_sync**: Back on topic: the led0_sync light flashes each time a sync pulse is generated. It lets you know your ROACH is alive.

- **led1_new_acc**: This lights up led1 each time a new accumulation is triggered.

- **led2_acc_clip**: This lights up led2 whenever clipping is detected.


There are also some [snap](https://casper.berkeley.edu/wiki/Snap) blocks, which capture data from the FPGA fabric and makes it accessible to the Power PC. This tutorial doesn't go into these blocks (in its current revision, at least), but if you have the inclination, have a look at their [documentation](https://casper.berkeley.edu/wiki/Snap).

If you've made it to here, congratulations, go and get yourself a cup of tea and a biscuit, then come back for part two, which explains the second part of the tutorial – actually getting the spectrometer running, and having a look at some spectra.

## Configuration and Control ##

### Hardware Configuration ###

The tutorial comes with a pre-compiled bof file, which is generated from the model you just went through (tut_spec.fpg).
All communication and configuration will be done by the python control script called tut_spec.py. 

Next, you need to set up your Skarab. Switch it on, making sure that:

•	Your tone source is set within the band of the ADC.  The default digital downconverter setting is 1 GHz, so signals within 375 MHz of this frequency should pass. In the tut_spec.py
script, 1 GHz is mapped to DC.  In our setup, we set the tone frequency to 1.054 GHz.  If you use a different tone frequency, be sure to update the ```freq_range_mhz = [0,80]``` in the Python script so that the plot covers the range of your tone.


### The tut_spec.py spectrometer script ###

Once you've got that done, it's time to run the script. First, check that you've connected the ADC to mega-array connector, and that the clock source is connected to clk_i of the ADC.
Now, if you're in linux, browse to where the tut3.py file is in a terminal and at the prompt type

```bash
 ./tut_spec.py <skarab IP or hostname> -l <accumulation length> -b <fpgfile name>
```

replacing <skarab IP or hostname> with the IP address of your Skarab, <accumulation length> is the number of accumulations, and <fpgfile name> with your fpgfile. You should see a spectrum like this:

![](../../_static/img/skarab/tut_spec/1p054GHz_sine_10accum.png)

Take some time to inspect the tut_spec.py script.  It is quite long, but don't be intimiated. Most of the script is configuration for the ADC.  The import lines begin after the ```#START OF MAIN``` comment.  There, you will see that the script

•	Instantiates the casperfpga connection with the Skarab

•	Uploads the fpg file

•	Sets the ADC

•	Records ADC snapshots, interleaves them and writes to a file adcN_data.txt where N is 0..4

•	Plots the spectral outputs of the memory blocks


## Conclusion ##
If you have followed this tutorial faithfully, you should now know:

•	What a spectrometer is and what the important parameters for astronomy are.

•	Which CASPER blocks you might want to use to make a spectrometer, and how to connect them up in Simulink.

•	How to connect to and control a Skarab spectrometer using python scripting.

