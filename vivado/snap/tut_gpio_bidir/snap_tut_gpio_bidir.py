#!/usr/bin/env python

'''
Script for testing the Bi-Directional GPIO Yellow Block created for CASPER Tutorial 7.
\n\n\
Author: Tyrone van Balla, January 2016
Reworked for SNAP and tested: Brian Bradford, May 2018
'''
import casperfpga
import time
import sys
import numpy as np

fpgfile = 'snap_tut_gpio_bidir.fpg'
fpgas = []

def exit_clean():
    try:
        for f in fpgas: f.stop()
    except:
        pass
    exit()

def exit_fail():
    print 'FAILURE DETECTED. Exiting . . .'
    exit()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("snap", help="<SNAP_HOSTNAME or IP>")
    parser.add_argument("-f", "--fpgfile", type=str, default=fpgfile, help="Specify the fpg file to load")
    parser.add_argument("-i", "--ipython", action='store_true', help="Enable iPython control")

    args = parser.parse_args()

    if args.snap == "":
        print 'Please specify a SNAP board. \nExiting'
        exit()
    else:
        snap = args.snap

    if args.fpgfile != '':
        fpgfile = args.fpgfile

print "Connecting to server %s . . . "%(snap),
fpga = casperfpga.CasperFpga(snap)
time.sleep(1)

if fpga.is_connected():
    print 'ok\n'
else:
    print 'ERROR connecting to server %s . . .'%(snap)
    exit_fail()

# program fpga with bitstream

print '------------------------'
print 'Programming FPGA...',
sys.stdout.flush()
fpga.upload_to_ram_and_program(fpgfile)
time.sleep(1)
print 'ok'

# intialize gpio bank control registers
fpga.write_int('a_is_input', 1)
fpga.write_int('b_is_input', 1)

if args.ipython:
    # open ipython session for manual testing of yellow block
    
    # list all registers first
    print '\nAvailable Registers:'
    registers = fpga.listdev()
    for reg in registers:
        if not('sys' in reg):
            print '\t',
            print reg
        else:
            pass
    print '\n'

    # how to use
    print 'Use "fpga" as the fpga object\n'

    import IPython; IPython.embed()
    
    print 'Exiting . . .'
    exit_clean()

'''
Automated testing of Bidirectional GPIO Block.
Sets one GPIO bank as output, other as input.
Writes to output bank, reads input.

Swaps mode of banks to demonstrate either bank can be either input or output.

'''

print '#################################'
# Send from GPIO_LED (B) to GPIO_GPIO (A) 
print '\nConfiguring to send from GPIO_LED (B) to GPIO_GPIO (A)\n'
fpga.write_int('a_is_input', 1) # GPIO_GPIO as input
fpga.write_int('b_is_input', 0) # GPIO_LED as output

print 'Initial Values: A: %s, B: %s\n' % (np.binary_repr(fpga.read_int('from_gpio_a'), width=4), np.binary_repr(fpga.read_int('from_gpio_b'), width=4))
print 'Writing 0xF to B . . . \n'

fpga.write_int('to_gpio_a', 0)  # dummy data written to GPIO_GPIO
fpga.write_int('to_gpio_b', 0xFFFF) # data written to GPIO_LED
time.sleep(0.01)

print 'A: 0 <------------- B: 0xF\n'

from_a = fpga.read_int('from_gpio_a') # read GPIO_GPIO
from_b = fpga.read_int('from_gpio_b') # read GPIO_LED

print 'Readback values: A: %s, B: %s\n' % (np.binary_repr(from_a, width=4), np.binary_repr(from_b, width=4))

print 'Writing 0x0 to B . . . \n'
print 'A: 0xF <---------- B: 0x0\n'

fpga.write_int('to_gpio_a', 0xFFFF) # dummy data written to GPIO_GPIO
fpga.write_int('to_gpio_b', 0x0) # data written to GPIO_LED
time.sleep(0.01)

from_a = fpga.read_int('from_gpio_a') # read GPIO_GPIO
from_b = fpga.read_int('from_gpio_b') # read GPIO_LED

print 'Readback values: A: %s, B: %s\n' % (np.binary_repr(from_a, width=4), np.binary_repr(from_b, width=4))

print '##################################'
# Send from GPIO_GPIO  (A) to GPIO_LED (B) 
print '\nConfiguring to send from GPIO_GPIO (A) to GPIO_LED (B)\n'
fpga.write_int('a_is_input', 0) # GPIO_GPIO as output
fpga.write_int('b_is_input', 1) # GPIO_LED as input

print 'Initial Values: A: %s, B: %s\n' % (np.binary_repr(fpga.read_int('from_gpio_a'), width=4), np.binary_repr(fpga.read_int('from_gpio_b'), width=4))
print 'Writing 0x0 to A . . . \n'

fpga.write_int('to_gpio_a', 0)  # data written to GPIO_GPIO
fpga.write_int('to_gpio_b', 0xFFFF) # dummy data written to GPIO_LED
time.sleep(0.01)

print 'A: 0 -------------> B: 0xF\n'

from_a = fpga.read_int('from_gpio_a') # read GPIO_GPIO
from_b = fpga.read_int('from_gpio_b') # read GPIO_LED

print 'Readback values: A: %s, B: %s\n' % (np.binary_repr(from_a, width=4), np.binary_repr(from_b, width=4))

print 'Writing 0xF to A . . . \n'

print 'A: 0xF ----------> B: 0x0\n'

fpga.write_int('to_gpio_a', 0xFFFF) # data written to GPIO_GPIO
fpga.write_int('to_gpio_b', 0x0) # dummy data written to GPIO_LED
time.sleep(0.01)

from_a = fpga.read_int('from_gpio_a') # read GPIO_GPIO
from_b = fpga.read_int('from_gpio_b') # read GPIO_LED

print 'Readback values: A: %s, B: %s\n' % (np.binary_repr(from_a, width=4), np.binary_repr(from_b, width=4))

exit_clean()
