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

# Installing Dependencies
## ROACH
For ROACH, you need Python 2.7 and python-pip. If you don't have these, you can probably install them with:

```bash
apt install python2.7
apt install python-pip
``` 

Once you have these, you can install all the dependencies you need with the following commands, run from the root directory of your repository (i.e., the `mlib_devel` directory):

```bash
# Install casperfpga dependencies
cd casperfpga
pip install -r requirements.txt

# Go back to the root of the repository
cd ..

# Install the requirements for your chosen platform
cd <your_platform_of_choice>/mlib_devel
pip install -r requirements.txt
```

## For non-ROACH platforms
For platforms newer than ROACH, you need Python 3 and python3-pip. If you don't have these, you can probably install them with:

```bash
apt install python3
apt install python3-pip
``` 
We thoroughly recommend using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) to separate the version of Python and its libraries the toolflow uses from the rest of your system. 

To create a Python 3 virtual environment:

```bash
# change directory to where you want the virtual environment to live
cd /home/user/work
# install virtualenv using pip3
sudo pip3 install virtualenv
# create a Python 3 virtual environment
virtualenv -p python3 casper_venv
# to activate the virtual environment:
source casper_venv/bin/activate
# to deactivate the virtual environment:
deactivate
```

Once you have these, you can install all the dependencies you need within your virtual environment with the following commands, run from the root directory of your repository (i.e., the `mlib_devel` directory):

```bash
# Activate your virtual environment
source /home/user/work/casper_venv/bin/activate

# Install casperfpga dependencies
cd casperfpga
pip3 install -r requirements.txt

# Go back to the root of the repository
cd ..

# Install the requirements for your chosen platform
cd <your_platform_of_choice>/mlib_devel
pip3 install -r requirements.txt
```


# Local Configuration

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
