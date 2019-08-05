# [CASPER Tutorials](http://casper-tutorials.readthedocs.io/en/latest/) [![Documentation Status](https://readthedocs.org/projects/casper-tutorials/badge/?version=latest)](https://casper-tutorials.readthedocs.io/en/latest/?badge=latest) #

These tutorials serve as an introduction to CASPER's [Toolflow](https://github.com/casper-astro/mlib_devel), [Software](https://github.com/casper-astro/casperfpga), and [Hardware](https://github.com/casper-astro/casper-hardware).

# Downloading

You can download these libraries by cloning this repository and initializing a `mlib_devel` library version appropriate for your hardware platform.

```bash
# Clone this repository from github
git clone https://github.com/casper-astro/tutorials_devel

# Go into the repository directory
cd tutorials_devel

# Download libraries for your chosen platform.
# <platform> should be one of: "roach2", "snap", "skarab", "red_pitaya"
# For example, to download the libraries for the SNAP board, you should run:
#
# ./activate_platform snap
#
./activate_platform <platform>
```

You will need a `startsg.local` script in your chosen platform directory (eg. `snap/startsg.local` for the SNAP board) before you can start the toolflow. See [The Toolflow Documentation](https://casper-toolflow.readthedocs.io/en/latest/src/Configuring-the-Toolflow.html#specifying-local-details) for details about what this script should contain.

Once you've downloaded the appropriate libraries, you can move to your chosen platform's directory and start the toolflow --

```bash
# Enter the directory for your chosen platform.
# Eg. for SNAP:
cd snap/

# Start the toolflow's MATLAB frontend
./startsg your.startsg.local.file
```

# Documentation
Documentation for these tutorials can be found [here](https://casper-tutorials.readthedocs.io/)
