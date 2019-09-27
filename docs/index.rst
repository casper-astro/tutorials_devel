CASPER Tutorials
============================================

Welcome to the CASPER tutorials page! Here you will find all the current tutorials for the SNAP, SKARAB and Red Pitaya platforms.

It is recommended to start with the introduction tutorial for the platform of your liking, then do that platform's GBE tutorial and finally move onto the spectrometer or correlator tutorial or the next difficulty tutorial.

Currently there are three hardware platforms supported through the CASPER Community:

1. SKARAB
2. SNAP
3. Red Pitaya

Environment setup
---------------------

The recommended OS is Ubuntu (currently 16.04) as it is what the majority of the collaboration are using. This makes it easier for us to support you. If you are so inclined, you could also use Red Hat, but we definitely do not support Windows. You are welcome to try but you will be on your own. You could always run Linux in a VM although this will increase your compile times.

The current compatibility matrix of software needed to run these tutorials is below:

(Note that official support for ROACH plaforms is no longer provided, however `this version <https://github.com/casper-astro/mlib_devel/tree/d77999047d2f0dc53e1c1e6e516e6ef3cdd45632/docs>`__ of ``mlib_devel`` contains all ROACH related documentation and `this version <https://github.com/casper-astro/tutorials_devel/tree/8bdd40d856ff542640d8f62a8d3029b084ae8efa/docs/tutorials/roach>`__ of ``tutorials_devel`` contains all ROACH tutorials)

+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|  Hardware      |   Operating System  |    Matlab Version  |    Xilinx Version  |    mlib_devel branch / commit   |   Python Version  |
+================+=====================+====================+====================+=================================+===================+
|SKARAB          | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|SNAP            | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+
|Red Pitaya      | Ubuntu 16.04        |  2018a             |  Vivado 2019.1.1   |  branch: `master`               |   Python 3        |
+----------------+---------------------+--------------------+--------------------+---------------------------------+-------------------+

Instructions on setting up the toolflow can be found `here <https://casper-toolflow.readthedocs.io/en/latest/index.html>`__.

Tutorial Instructions
----------------------

If you are new to astronomy signal processing, here is `Tutorial 0: some basic introduction into astronomy signal processing. <https://github.com/SparkePei/tut0>`__ If you already have a lot of experience on it, you can go directly to the introduction tutorials below for CASPER FPGA design and implementation.

If you are a beginner, we recommend the Step-by-Step tutorials, however if you should get stuck, prefer a less tedious method of learning, or already have decent feel for these tools, links to Completed tutorials are available with commented models.

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
4. Spectrometer Tutorial :doc:`Step-by-Step <tutorials/skarab/tut_spec>` or `Completed <https://github.com/casper-astro/tutorials_devel/tree/workshop2019/vivado_2018/skarab/tut_spec>`__

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: SKARAB Tutorials

   tutorials/skarab/tut_intro
   tutorials/skarab/tut_40gbe
   tutorials/skarab/tut_hmc
   tutorials/skarab/tut_spec  
   
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

..  toctree::
    :hidden:
    :maxdepth: 1
    :caption: Documentation

    CASPER Documentation <https://casper-toolflow.readthedocs.io/en/latest/index.html>
    Block Documentation <https://casper-toolflow.readthedocs.io/en/latest/blockdocumentation.html>
    The CASPER Toolflow <https://casper-toolflow.readthedocs.io/en/latest/jasper_documentation.html>
    Toolflow Sourcecode <https://casper-toolflow.readthedocs.io/en/latest/src/jasper_library_modules/modules.html>
    casperfpga Sourcecode <https://casper-toolflow.readthedocs.io/projects/casperfpga/en/latest/>



