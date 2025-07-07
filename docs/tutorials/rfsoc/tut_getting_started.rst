
Getting Started With RFSoC
==========================

Introduction
------------

This tutorial presents the steps to setup the development environment for using
the CASPER tools to target supported RFSoC platforms.

Environment Setup
-----------------

Pre requisites
^^^^^^^^^^^^^^

The recommended environment setup and software required is more or less
consistent with the standard CASPER
`setup <https://casper-toolflow.readthedocs.io/projects/tutorials/en/latest/#environment-setup]>`_.
In this case, for RFSoC what you will need is:


* Compatible Linux host operating system (tested on RHEL 7.9, 8.4 and Ubuntu 18.04 LTS, 20.04 LTS)
* Vivado 2021.1
* Matlab 2021a (with Simulink)
* Python 3 environment 
* Development branches of the CASPER "toolflow" ``mlib_devel`` `library <https://gitlab.ras.byu.edu/alpaca/casper/mlib_devel/-/tree/rfsocs/zcu216>`_ and board
  communication ``casperfpga`` `library <https://gitlab.ras.byu.edu/alpaca/casper/casperfpga/-/tree/rfsocs/rfdc>`_ with RFSoC support
   * These links need to be fixed. Possibly link `here <https://github.com/casper-astro/mlib_devel>`_ instead? Currently links to gitlab. Other possible links include `the main casper library <https://gitlab.ras.byu.edu/alpaca/casper/mlib_devel/-/tree/m2021a/casper_library>`_ and `the main board support library <https://gitlab.ras.byu.edu/alpaca/casper/mlib_devel/-/tree/m2021a/xps_library>`_.
* `Xilinx device tree repository <https://github.com/Xilinx/device-tree-xlnx/>`_

Some help and pointers for general toolflow and software installation can be
found `here <https://casper-toolflow.readthedocs.io/en/latest/src/Installing-the-Toolflow.html#pre-requisites>`_. It is strongly recommended to use an
isolated runtime environment of python. This is done with a virtual environment
using the ``venv`` `python package <https://docs.python.org/3/tutorial/venv.html>`_ or some other tool with an
environment manager such as `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_. The following assumes an
OS and required vendor software has been installed.

Core Setup
----------

With a compatible Linux OS, Vivado, and Matlab installed, we will now complete
the following three core tasks:

#. Prepare and install the core "toolfow" ``mlib_devel``
#. Prepare and setup of the CASPER platform (usually the fun part)
#. Prepare and install the communication library ``casperfpga``

To begin acquiring the toolflow, start within a new python environment by fetching the development
branches and dependencies needed to work with RFSoC. The following will assume a
working directory called ``sandbox`` just as an isolated example through this
tutorial. Feel free to organize suitable to your preferences.

Toolflow Setup
^^^^^^^^^^^^^^

.. code-block:: bash

   $ cd </path/to>/sandbox
   $ mkdir casper
   $ cd casper
   $ git clone https://github.com/casper-astro/mlib_devel.git
   $ cd mlib_devel
   $ git checkout m2021a origin/m2021a

   # install pacakge dependencies
   $ pip install -r requirements.txt

   # initialize submodules
   $ git submodule init
   $ git submodule update

   # fetch a copy of the xilinx device tree repo
   $ cd </path/to>/sandbox
   $ mkdir xilinx
   $ cd xilinx
   $ git clone https://github.com/xilinx/device-tree-xlnx.git

   # update or create your `startsg.local` config file 
   $ cd </path/to>/sandbox/casper/mlib_devel
   $ cp startsg.local.example ./startsg.local

   # with you favorite text editor open `startsg.local` and update the following
   # environment variables
   XILINX_PATH=</path/to/your/Xilinx>/Vivado/2021.1
   MATLAB_PATH=</path/to/your/Matlab>/R2021a
   COMPOSER_PATH=</path/to/your/Xilinx>/Model_Composer/2021.1
   JASPER_BACKEND=vitis
   XLNX_DT_REPO_PATH=</path/to>/sandbox/xilinx/device-tree-xlnx

   # The following is an example of my startsg.local
   export XILINX_PATH=/opt/Xilinx/Vivado/2021.1 
   export MATLAB_PATH=/opt/MATLAB/R2021a
   export COMPOSER_PATH=/opt/Xilinx/Model_Composer/2021.1
   export PLATFORM=lin64 
   export JASPER_BACKEND=vitis
   export XLNX_DT_REPO_PATH=/home/mcb/git/xilinx/device-tree-xlnx

   # Start Matlab and System Generator
   $ ./startsg

Platform Processor System Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These steps are generally platform agnostic as this focuses more on preparing
and booting the processor system (PS). However, there are some platform
dependent hardware setup and procedures to be discussed later.
`Download <https://casper.groups.et.byu.net>`_ the casperized image for your target RFSoC board and
locate a 16 GB micro SD card. We next start to unpack and flash the image. The
following uses the ``zcu216_casper.img`` as an example, the target ``.img`` download
file would be replaced in all subsequent commands.

.. code-block:: bash

   # navigate to the download location of the compressed tar and unpack it
   $ cd </path/to/downloads>
   $ tar -xzf zcu216_casper.img.tar.gz

   # the full uncompressed image `zcu216_casper.img` is now in the current directory
   $ ls zcu216_casper.*
   zcu216_casper.img  zcu216_casper.img.tar.gz

   # plug in the micro sd card, on OS's like Ubuntu the disk may auto mount,
   # unmount before preceeding.

   # Take note of the kernel registered block device
   # such as `sdb, sdc, sdd, etc.`. This can be done with the `dmesg` utility e.g.,
   $ dmesg
   108821.527053] scsi host38: usb-storage 2-2:1.0
   [108822.527801] scsi 38:0:0:0: Direct-Access     TS-RDF5  SD  Transcend    TS38 PQ: 0 ANSI: 6
   [108822.528460] sd 38:0:0:0: Attached scsi generic sg3 type 0
   [108822.829512] sd 38:0:0:0: [sdd] 31116288 512-byte logical blocks: (15.9 GB/14.8 GiB)
   [108822.830188] sd 38:0:0:0: [sdd] Write Protect is off
   [108822.830197] sd 38:0:0:0: [sdd] Mode Sense: 23 00 00 00
   [108822.830867] sd 38:0:0:0: [sdd] Write cache: disabled, read cache: enabled, doesnt support DPO or FUA
   [108822.835071]  sdd: sdd1 sdd2
   [108822.837460] sd 38:0:0:0: [sdd] Attached SCSI removable disk
   [109641.322489]  sdd: sdd1 sdd2

   # in this example the sd card block device is `sdd`

   # flash the sd card with the `dd` utility, wait until this completes. It can take awhile
   # as we must wait to sync all the I/O, must also have root access
   $ sudo dd if=zcu216_casper.img of=/dev/sdd bs=32MB

Take out the SD card and plug it into your platform board. Place the DIP
switches that select the boot device to SD mode. You are about ready to power-on
the board.

Prior to booting the board, provide a connection to the 1GBE port and
review the `Network Configuration Section <#platform-network-configuration>`_ to
understand how communication will be established on the board. A micro-USB
serial cable can be optionally attached and the serial output from the processor
can be monitored with a utility such as ``minicom``. The serial port is configured
for baud 115200, 8 data bits, no parity, 1 stop bit. This output can be helpful
to obtain the IP address if there is no direct access to configure a DHCP server
or a static IP address was not set before hand.

Power-on the board. As long as the IP address of the board is known there is no
requirement at this time to log in. The image comes pre-configured to be ready
to interface with ``casperfpga``. In this case, if the IP is known the last step
is to `install ``casperfpga`` <#setup-casperfpga>`_ and test commmunications.
Otherwise, using the serial connection login with the default user ``casper`` and
default password ``casper`` and run the ``ip addr`` command to learn the IP address
of your board.

Setup Casperfpga
^^^^^^^^^^^^^^^^

Next is to install ``casperfpga``. The same Python 3 environment can be used to keep
it simple.

.. code-block:: bash

   $ cd </path/to>/sanbox/casper
   $ git clone https://github.com/casper-astro/casperfpga.git
   $ cd casperfpga
   $ git checkout py38 origin/py38

   # install package requirements
   $ pip install -r requirements.txt

   # build and install `casperfpga`
   $ pip install .

``casperfpga`` is now installed and we can test connection with the platform. To
do this we can run a few commands in IPython. First, change out of the
``casperfpga`` directory as we want to reference the package we just installed
instead of the one in the ``casperfpga`` source directory.

Start an IPython session; In this example the ZCU216 IP address was assigned
to ``192.168.2.101``.

.. code-block:: python

   In [1]: import casperfpga

   In [2]: fpga = casperfpga.CasperFpga('192.168.2.101')

   In [3]: fpga.is_connected()
   Out[3]: True

This does not seem like an incredibly exciting result, but everything is setup
and we are now ready to move on to testing the toolflow installation and get more
familiar with your platform image and ``casperfpga`` in the `next
tutorial <./tut_platform.md>`_.

Misc. Configuration
-------------------

Platform Network Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each platform image is configured by default to use DHCP to receive an IP
address when the kernel boots. For the ZCU216/208, ZCU111, and RFSoC2x2
platforms the first stage boot loader is configured to look at the EEPROM for a
MAC address, if a valid address is not found then a randomly generated one is
created at each boot. Random MAC generation or setting a static IP can be
overridden by either `manually writing <#manually-writing-platform-mac-address>`_
a valid MAC to the EEPROM or using the Linux kernel's Network Manager
configuration scripts. The HTG ZRF16-29/49DR boards boot with the static MAC
address ``0a:4c:50:14:42:00`` again, this can be overridden by using a Network
Manager configuration script.

Manually Writing Platform MAC Address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The on board EEPROMs are interfaced over i2c. They can be programmed with the first stage
boot loader's (U-Boot) i2c utility, with a Linux i2c utility or custom userspace
application, and some boards will expose i2c header pins to attach a serial
programmer. As setting the MAC address in the EEPROM is a "set once and forget
about" type of thing, a quick an easy way is to use U-boot's i2c utility.

With a micro-USB serial cable connected to the board begin to monitor the serial
output from the processor. Power-on the board and the serial console will begin
to display boot progress starting with the first stage boot loader. After
reporting status of peripheral hardware the prompt "hit any key to stop
autoboot:". Before the count down ends, interrupt with the keyboard starting the
U-Boot command line interface. The output would have been similar to the
following:

.. code-block:: bash

   Xilinx Zynq MP First Stage Boot Loader 
   Release 2020.2   Jul 15 2021  -  16:48:09
   NOTICE:  ATF running on XCZU49DR/silicon v4/RTL5.1 at 0xfffea000
   NOTICE:  BL31: v2.2(release):xilinx_rebase_v2.2_2020.1-10-ge6eea88b1
   NOTICE:  BL31: Built : 16:45:03, Jul 15 2021


   U-Boot 2020.01 (Jul 15 2021 - 16:49:01 +0000)

   Model: ZynqMP ZCU216 RevA
   Board: Xilinx ZynqMP
   DRAM:  4 GiB
   PMUFW:  v1.1
   EL Level:       EL2
   Chip ID:        zu49dr
   NAND:  0 MiB
   MMC:   mmc@ff170000: 0
   In:    serial@ff000000
   Out:   serial@ff000000
   Err:   serial@ff000000
   Bootmode: LVL_SHFT_SD_MODE1
   Reset reason:   SOFT 
   Net:   
   ZYNQ GEM: ff0e0000, mdio bus ff0e0000, phyaddr 12, interface rgmii-id

   Warning: ethernet@ff0e0000 (eth0) using random MAC address - 3a:b0:c7:80:96:3f
   eth0: ethernet@ff0e0000
   Hit any key to stop autoboot:  0 
   ZynqMP>

Notice the 'Warning' line informing that a random MAC address was created. We
now begin to peek and poke using the i2c utility.

.. code-block:: bash

   # get information from the i2c bus, look for the "eeprom" node
   # on the zcu216 this is at address 54
   ZynqMP> i2c bus
   .
   .
   Bus 2:  i2c@ff030000->i2c-mux@74->i2c@0  (active 2)
      54: eeprom@54, offset len 2, flags 0
   .
   .

   # target that bus
   ZynqMP> i2c dev 2
   Setting bus to 2

   # we can get help on what the i2c utility can do
   ZynqMP> i2c    
   i2c - I2C sub-system

   Usage:
   i2c bus [muxtype:muxaddr:muxchannel] - show I2C bus info
   i2c crc32 chip address[.0, .1, .2] count - compute CRC32 checksum
   i2c dev [dev] - show or set current I2C bus
   i2c loop chip address[.0, .1, .2] [# of objects] - looping read of device
   i2c md chip address[.0, .1, .2] [# of objects] - read from I2C deviceA
   i2c mm chip address[.0, .1, .2] - write to I2C device (auto-incrementing)
   .
   .
   .

   # We will only need to read/write here, we can start by taking a peek at the
   # first 16 bytes of the memory using the address reported by `i2c bus`.
   # Depending on the platform this could be initialized or not, in the case of the
   # ZCU216 and ZCU111 it is.
   ZynqMP> i2c md 0x54 0x0 0x10
   0000: 5a 43 55 32 31 36 ff ff 11 ff ff ff 99 ff ff ff    ZCU216..........

   # The MAC address is stored as 6 bytes at offset 0x20 in the eeprom. First write
   # the address we want to place in the eeprom inside U-Boot's scratchpad area in
   # DDR memory
   ZynqMP> mm.b 0x 
   00000000: 00 ? 0a
   00000001: 00 ? 4c 
   00000002: 00 ? 50
   00000003: 00 ? 41 
   00000004: 00 ? 43
   00000005: 00 ? 41
   00000006: 00 ? q

   # now write from address 0x0 to the eeprom at address 0x20 and write those 6 bytes
   ZynqMP> i2c write 0x00 0x54 0x20 0x6

   # read back to make sure it worked as expected
   ZynqMP> i2c md 0x54 0x20 0x6        
   0020: 0a 4c 50 41 43 41    .LPACA

   # reboot the board
   ZynqMP> reset

   # the first stage boot loader will start back up, reporting the same information
   # as before, but this time the `Warning` should now read
   .
   .
   Warning: ethernet@ff0e0000 using MAC address from ROM
   .
   .

The MAC address has been set and you can let the auto boot counter timeout and
proceed to boot.
