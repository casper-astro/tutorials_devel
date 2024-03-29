# RFSoC 4x2

vendor links: [PYNQ infromation][pynq-rfsoc4x2] and [Real-Digital][real-digital-rfsoc4x2]

![](../../../_static/img/rfsoc/readme/rfsoc4x2.jpeg)

# RF Clocking

The following figure shows a high-level block diagram for the clocking network:

At boot the RFSoC 4x2 the LMK will be programmed to provide a 122.88 MHz
reference to the PL and a 245.76 MHz reference for the ADC and DAC LMX. The LMX
PLLs are programmed to provide a reference of 491.52 reference to the RFDC ADC
and DAC tiles.

[pynq-rfsoc4x2]: https://www.rfsoc-pynq.io
[real-digital-rfsoc4x2]: https://www.realdigital.org/hardware/rfsoc-4x2

