CASPER Tutorials
============================================

Welcome to the `CASPER tutorials page <https://casper-tutorials.readthedocs.io/en/latest/>`__! Here you will find all the current tutorials for the SNAP, SKARAB and Red Pitaya platforms.

The tutorial repository can be found on `github <https://github.com/casper-astro/tutorials_devel.git>`__ and can be downloaded with the command ``git clone https://github.com/casper-astro/tutorials_devel.git``.

It is recommended to start with the introduction tutorial for the platform of your liking, then do that platform's GBE tutorial and finally move onto the spectrometer or correlator tutorial or the next difficulty tutorial.

Currently there are tutorials available for five hardware platforms within the CASPER community:

1. ROACH
2. ROACH2
3. SKARAB
4. SNAP
5. Red Pitaya

Environment setup
---------------------

The recommended OS is Ubuntu (currently 16.04) as it is what the majority of the collaboration are using. This makes it easier for us to support you. If you are so inclined, you could also use Red Hat, but we definitely do not support Windows. You are welcome to try but you will be on your own. You could always run Linux in a VM although this will increase your compile times.

The current compatibility matrix of software needed to run these tutorials is below:

(Note that official support for ROACH plaforms is no longer provided, however `this version <https://github.com/casper-astro/mlib_devel/tree/d77999047d2f0dc53e1c1e6e516e6ef3cdd45632/docs>`__ of `mlib_devel` contains all ROACH related documentation and ROACH tutorials can be found `here <https://casper-tutorials.readthedocs.io/en/latest/tutorials/roach/tut_intro.html>`__)

+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|  Hardware      |   Operating System  |    Matlab Version  |    Xilinx Version  |    mlib_devel branch / commit   |   Python Version  |
+================+=====================+====================+====================+=================================+===================+
|ROACH1/2        | Ubuntu 14.04        |  2013b             |  ISE 14.7          |  branch: `roach`                |   Python 2.7      |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|SKARAB          | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|SNAP            | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|Red Pitaya      | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+

Instructions on setting up an environment in which to run these tutorials can be found `here <https://github.com/casper-astro/tutorials_devel/blob/workshop2019/README.md>`__. Instructions on setting up the toolflow-proper can be found `here <https://casper-toolflow.readthedocs.io/en/latest/index.html>`__.

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

1. Introduction Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_intro>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/skarab/tut_intro>`__
2. 40GbE Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_40gbe>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/skarab/tut_40gbe>`__
3. HMC Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_hmc>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/skarab/tut_hmc>`__
4. Spectrometer Tutorials :doc:`Step-by-Step <tutorials/skarab/tut_spec/tut_spec_index>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/skarab/tut_spec>`__
5. :doc:`ADC Synchronous Data Acquisition Tutorials <tutorials/skarab/tut_adc/tut_adc_index>`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: SKARAB Tutorials

   tutorials/skarab/tut_intro
   tutorials/skarab/tut_40gbe
   tutorials/skarab/tut_hmc
   tutorials/skarab/tut_spec/tut_spec_index  
   tutorials/skarab/tut_adc/tut_adc_index
   
**Red Pitaya**

1. :doc:`Guide to Setting Up Your New Red Pitaya <tutorials/redpitaya/red_pitaya_setup>` 
2. Introduction Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_intro>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/red_pitaya/tut_intro>`__
3. ADC and DAC Interface Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_adc_dac>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/red_pitaya/tut_adc_dac>`__
4. Spectrometer Tutorial :doc:`Step-by-Step <tutorials/redpitaya/tut_spec>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/master/red_pitaya/tut_spec>`__

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Red Pitaya Tutorials

   tutorials/redpitaya/red_pitaya_setup 
   tutorials/redpitaya/tut_intro
   tutorials/redpitaya/tut_adc_dac
   tutorials/redpitaya/tut_spec

..  toctree::
    :hidden:
    :maxdepth: 1
    :caption: Documentation

    CASPER Documentation <https://casper-toolflow.readthedocs.io/en/latest/index.html>
    Block Documentation <https://casper-toolflow.readthedocs.io/en/latest/blockdocumentation.html>
    The CASPER Toolflow <https://casper-toolflow.readthedocs.io/en/latest/jasper_documentation.html>
    Toolflow Sourcecode <https://casper-toolflow.readthedocs.io/en/latest/src/jasper_library_modules/modules.html>
    casperfpga Sourcecode <https://casper-toolflow.readthedocs.io/projects/casperfpga/en/latest/>

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



