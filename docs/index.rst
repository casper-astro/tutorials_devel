CASPER Tutorials
============================================

Welcome to the CASPER tutorials page! Here you will find all the current tutorials for the ROACH, SNAP, SKARAB and Red Pitaya platforms.

It is recommended to start with the introduction tutorial for the platform of your liking, then do that platform's GBE tutorial and finally move onto the spectrometer or correlator tutorial or the next difficulty tutorial.

Currently there are five hardware platforms supported through the CASPER Community:

1. ROACH
2. ROACH2
3. SKARAB
4. SNAP
5. Red Pitaya

It is worth noting that even though SNAP, SKARAB and Red Pitaya require their firmwares to be developed using Xilinx's Vivado (as opposed to ISE), the **SNAP** tutorials are **very** similar to the ROACH/2 tutorials. In fact, the only real difference is the choice of hardware platform that is made in Simulink. This is done by selecting the **SNAP** Yellow Block in the Simulink library under *CASPER XPS Blockset -> Hardware Platforms*

Tutorial Instructions
----------------------

If you are new to astronomy signal processing, here is `Tutorial 0: some basic introduction into astronomy signal processing. <https://github.com/SparkePei/tut0>`__ If you already have a lot of experience on it, you can go directly to the introduction tutorials below for CASPER FPGA design and implementation.

If you are a beginner, we recommend the Step-by-Step tutorials, however if you should get stuck, prefer a less tedious method of learning, or already have decent feel for these tools, links to Completed tutorials are available with commented models.

Vivado
^^^^^^^^^

**SNAP**

1. Introduction Tutorial: :doc:`Step-by-Step <tutorials/snap/tut_intro>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/vivado/snap/tut_intro>`_
2. 10GbE Tutorial: :doc:`Step-by-Step <tutorials/snap/tut_ten_gbe>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/vivado/snap/tut_tge>`__
3. Spectrometer Tutorial :doc:`Step-by-Step <tutorials/snap/tut_spec>` or  `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/vivado/snap/tut_spec>`__
4. Correlator Tutorial :doc:`Step-by-Step <tutorials/snap/tut_corr>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/vivado/snap/tut_corr>`__
5. Yellow Block Tutorial: :doc:`Bidirectional GPIO <tutorials/snap/tut_gpio_bidir>` 

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: SNAP Tutorials

   tutorials/snap/tut_intro
   tutorials/snap/tut_ten_gbe
   tutorials/snap/tut_spec
   tutorials/snap/tut_corr
   tutorials/snap/tut_gpio_bidir

**SKARAB**

1. Introduction Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_intro>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/skarab/tut_intro>`__
2. 40GbE Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_40gbe>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/skarab/tut_40gbe>`__
3. HMC Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_hmc>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/skarab/tut_hmc>`__

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: SKARAB Tutorials

   tutorials/skarab/tut_intro
   tutorials/skarab/tut_40gbe
   tutorials/skarab/tut_hmc  
   
**Red Pitaya**

1. Introduction Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_intro>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/red_pitaya/tut_intro>`__
2. ADC and DAC Interface Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_adc_dac>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/red_pitaya/tut_adc_dac>`__
3. Spectrometer Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_spec>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/red_pitaya/tut_spec>`__

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Red Pitaya Tutorials

   tutorials/redpitaya/tut_intro
   tutorials/redpitaya/tut_adc_dac
   tutorials/redpitaya/tut_spec

ISE
^^^^^

**ROACH1/2**

1. :doc:`Introduction Tutorial <tutorials/roach/tut_intro>`
2. :doc:`10GbE Tutorial <tutorials/roach/tut_ten_gbe>`
3. :doc:`Spectrometer Tutorial <tutorials/roach/tut_spec>`
4. :doc:`Correlator Tutorial <tutorials/roach/tut_corr>`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: ROACH Tutorials

   tutorials/roach/tut_intro
   tutorials/roach/tut_ten_gbe
   tutorials/roach/tut_spec
   tutorials/roach/tut_corr

Environment setup
---------------------

OS
^^^^

It is recommended to use Ubuntu 14.04. 16.04 has also been known to work, although the setup process can be a bit of a headache. Ubuntu 16.04 LTS will work with SKARAB and Red Pitaya.

Matlab and Xilinx
^^^^^^^^^^^^^^^^^^

To use the tutorials you will need to install the versions of Matlab and the Xilinx tools particular to the hardware you plan to use. See the installation matrix below.

============  ==================  ==================
**Hardware**  **Matlab Version**  **Xilinx Version**
============  ==================  ==================
ROACH1/2      2013b               ISE 14.7 
SKARAB        2018a               Vivado 2018.2
SNAP          2016b               Vivado 2016.4 
Red Pitaya    2018a               Vivado 2018.2 
============  ==================  ================== 

Modifications to be run after installs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**ROACH1/2**

Xilinx removed support for several hardware pcores we use for ROACH1/2 from ISE 14. So the current solution is to add the following pcores from the Xilinx 11 install to your `XPS_ROACH_BASE/pcores` folder or to your 14 install directory at `Xilinx/14.7/ISE_DS/EDK/hw/XilinxProcessorIPLib/pcore`.

`OPB pcores <https://www.dropbox.com/s/eq57n5td37yrwma/pcores_for_ise13.zip?dl=1>`__

* bram_if_cntlr_v1_00_a
* bram_if_cntlr_v1_00_b
* ipif_common_v1_00_c
* opb_arbiter_v1_02_e
* opb_bram_if_cntlr_v1_00_a
* opb_ipif_v3_00_a
* opb_opb_lite_v1_00_a
* opb_v20_v1_10_c
* proc_common_v1_00_a

**All installs**

The syntax in the Xilinx Perl scripts is not supported under the Ubuntu default shell Dash.
Change the symbolic link sh -> dash to sh -> bash:
::

    cd /bin/
    sudo rm sh
    sudo ln -s bash sh


Point gmake to make by creating the symbolic link gmake -> make:
::

    cd /usr/bin/
    sudo ln -s make gmake


If you are not getting any blocks in Simulink (Only seen in CentOS)
change the permissions on /tmp/LibraryBrowser to a+rwx:
::

    chmod a+rwx /tmp/LibraryBrowser
