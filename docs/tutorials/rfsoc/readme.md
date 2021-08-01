# CASPER RFSoC

# Introduction
The Xilinx Zynq UltraScale+ RFSoC integrates programmable logic with the Zynq
ARM (A53) processor, high speed serial transceivers, and several
multi-gigasample per second RF Data Converters (RFDC) capable of digital down
conversion and direct sampling for RF signals up to 6 GHz. These features are
ideal for small form factor and low power digitizers streaming raw voltages over
2x100GbE, or suitable as modest sized channelizers using the available fabric
resources. This documentation aims to introduce RFSoC to the CASPER community.

The primary IP that is the focus of the RFSoC is the RF-Data Converter (RFDC).
with the recent development of the RFDC yellow block and several newly supported
RFSoC CASPER platforms.

zcu216, zcu208, zcu111, zrf16 (both 29/49DR rev.), and PYNQ
2x2. The current capabilities of each platform, how the RFDC integrates with the
tool flow, current software (i.e., casperfpga) support, available tutorials and
resources, and future development roadmap will also be shown. The addition of
these new CASPER platforms and the extent to which the RFDC yellow block
integrates with the hardware to provide flexible design configurations will
continue to proliferate the design philosophy of CASPER of decreasing the
time-to-science metric and provide a way of bringing the needed capabilities to
next generation instruments.

![](../../_static/img/rfsoc/readme/rfsoc_spec_table.png)

There are several family generations of the

# Platforms
The CASPER library contains support for the RFDC are 5 RFSoC platforms that have been tested
  * [ZCU216][zcu216]
  * [ZCU111][zcu111]
  * [PYNQ RFSoC 2x2][pynq-rfsoc2x2]
  * [HTG ZRF16-29DR][htg-zrf16] [\*\*][htg-disclaimers]
  * [HTG ZRF16-49DR][htg-zrf16] [\*\*][htg-disclaimers]


# Tutorials
* [Getting Started][getting-started]
* [Platform and Simulink Overview][platform-overview]
* [Using the RFDC][rfdc]

## Designs
* [Example Designs][example-designs]

[zcu216]: https://www.xilinx.com/products/boards-and-kits/zcu216.html
[zcu208]: https://www.xilinx.com/products/boards-and-kits/zcu208.html
[zcu111]: https://www.xilinx.com/products/boards-and-kits/zcu111.html
[htg-zrf16]: http://www.hitechglobal.com/Boards/16ADC-DAC_Zynq_RFSOC.htm
[pynq-rfsoc2x2]: https://www.rfsoc-pynq.io
[htg-disclaimers]: ./htg-disclaimers.md


[getting-started]: ./tut_getting_started.md
[platform-overview]: ./tut_platform.md
[rfdc]: ./tut_rfdc.md
[example-designs]: ./tut_designs/readme.md
