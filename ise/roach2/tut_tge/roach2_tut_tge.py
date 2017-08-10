#!/usr/bin/env python

'''
This script demonstrates programming an FPGA, configuring 10GbE cores and checking transmitted and received data using the Python KATCP library along with the katcp_wrapper distributed in the casperfpga package.
\n\n 
Author: Jason Manley, August 2009.
Updated by: Tyrone van Balla, December 2015
Updated for CASPER 2016 workshop. Updated to use casperfpga library and for ROACH-2
'''
import casperfpga
import time
import struct
import sys
import logging
import socket

#Decide where we're going to send the data, and from which addresses:
# IP addr, ports & macs should match addresses specified in gbe blocks in simulink model
dest_ip  =192*(2**24) + 168*(2**16) + 5*(2**8) + 16
fabric_port=10000         
source_ip= 192*(2**24) + 168*(2**16) + 5*(2**8) + 20
mac_base=20015998304256

pkt_period = 16384  #how often to send another packet in FPGA clocks (200MHz)
payload_len = 128   #how big to make each packet in 64bit words

brams=['bram_msb','bram_lsb','bram_oob']
tx_snap = 'snap_gbe0_tx'
rx_snap = 'snap_gbe1_rx'

tx_core_name = 'gbe0'
rx_core_name = 'gbe1'

fpgfile = 'tut2.fpg'
fpgas=[]

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    exit()

def exit_clean():
    try:
        for f in fpgas: f.stop()
    except: pass
    exit()

# debug log handler
class DebugLogHandler(logging.Handler):
    """A logger for KATCP tests."""

    def __init__(self,max_len=100):
        """Create a TestLogHandler.
            @param max_len Integer: The maximum number of log entries
                                    to store. After this, will wrap.
        """
        logging.Handler.__init__(self)
        self._max_len = max_len
        self._records = []

    def emit(self, record):
        """Handle the arrival of a log message."""
        if len(self._records) >= self._max_len: self._records.pop(0)
        self._records.append(record)

    def clear(self):
        """Clear the list of remembered logs."""
        self._records = []

    def setMaxLen(self,max_len):
        self._max_len=max_len

    def printMessages(self):
        for i in self._records:
            if i.exc_info:
                print '%s: %s Exception: '%(i.name,i.msg),i.exc_info[0:-1]
            else:    
                if i.levelno < logging.WARNING: 
                    print '%s: %s'%(i.name,i.msg)
                elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                    print '%s: %s'%(i.name,i.msg)
                elif i.levelno >= logging.ERROR: 
                    print '%s: %s'%(i.name,i.msg)
                else:
                    print '%s: %s'%(i.name,i.msg)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("roach", help="<ROACH_HOSTNAME or IP>")
    parser.add_argument("--noprogram", action='store_true', help="Don't program the fpga")
    parser.add_argument("-s", "--silent", action='store_true', help="Don't print the contents of the packets")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot the TX and RX counters. Requires matplotlib/pylab")
    parser.add_argument("-a", "--arp", action='store_true', help="Print the ARP table and other info")
    parser.add_argument("-f", "--fpgfile", type=str, default=fpgfile, help="Specify the fpg file to load")

    args = parser.parse_args()

    if args.roach=="":
        print 'Please specify a ROACH board. \nExiting.'
        exit()
    else:
        roach = args.roach

    if args.fpgfile != '':
        fpgfile = args.fpgfile
try:
    lh = DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(roach)),
    fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(roach)
        exit_fail()
    
    if not args.noprogram:
        print '------------------------'
        print 'Programming FPGA...',
        sys.stdout.flush()
        fpga.upload_to_ram_and_program(fpgfile)
        time.sleep(10)
        print 'ok'

    print '---------------------------'    
    print 'Disabling output...',
    sys.stdout.flush()
    fpga.write_int('pkt_sim_enable', 0)
    print 'done'

    print '---------------------------'    
    print 'Port 0 linkup: ',
    sys.stdout.flush()
    gbe0_link=bool(fpga.read_int('gbe0_linkup'))
    print gbe0_link
    if not gbe0_link:
        print 'There is no cable plugged into port0. Please plug a cable between ports 0 and 1 to continue demo. Exiting.'
        exit_clean()
    print 'Port 1 linkup: ',
    sys.stdout.flush()
    gbe1_link=bool(fpga.read_int('gbe1_linkup'))
    print gbe1_link
    if not gbe1_link:
        print 'There is no cable plugged into port1. Please plug a cable between ports 0 and 1 to continue demo. Exiting.'
        exit_clean()

    print '---------------------------'
    print 'Configuring receiver core...',
    sys.stdout.flush()
    fpga.tengbes.gbe0.tap_start(restart=False)
    print 'done'
    print 'Configuring transmitter core...',
    sys.stdout.flush()
    fpga.tengbes.gbe1.tap_start(restart=False)
    print 'done'

    print '---------------------------'
    print 'Setting-up packet source...',
    sys.stdout.flush()
    fpga.write_int('pkt_sim_period',pkt_period)
    fpga.write_int('pkt_sim_payload_len',payload_len)
    print 'done'

    print 'Setting-up destination addresses...',
    sys.stdout.flush()
    fpga.write_int('dest_ip',dest_ip)
    fpga.write_int('dest_port',fabric_port)
    print 'done'

    print 'Resetting cores and counters...',
    sys.stdout.flush()
    fpga.write_int('rst', 3)
    fpga.write_int('rst', 0)
    print 'done'

    time.sleep(2)

    if args.arp:
        print '\n\n==============================='
        print '10GbE Transmitter core details:'
        print '==============================='
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        fpga.tengbes.gbe0.print_10gbe_core_details(arp=True)
        print '\n\n============================'
        print '10GbE Receiver core details:'
        print '============================'
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        fpga.tengbes.gbe1.print_10gbe_core_details(arp=True)

    print 'Sent %i packets already.'%fpga.read_int('gbe0_tx_cnt')
    print 'Received %i packets already.'%fpga.read_int('gbe1_rx_frame_cnt')

    print '------------------------'
    print 'Triggering snap captures...',
    sys.stdout.flush()
    fpga.write_int(tx_snap+'_ctrl',0)
    fpga.write_int(rx_snap+'_ctrl',0)
    fpga.write_int(tx_snap+'_ctrl',1)
    fpga.write_int(rx_snap+'_ctrl',1)
    print 'done'

    print 'Enabling output...',
    sys.stdout.flush()
    fpga.write_int('pkt_sim_enable', 1)
    print 'done'

    time.sleep(2)

    tx_size = fpga.read_int(tx_snap+'_addr')+1
    rx_size = fpga.read_int(rx_snap+'_addr')+1
    if tx_size <= 1:
        print ('ERR: Not transmitting anything. This should not happen. Exiting.')
        exit_clean()
    if rx_size <= 1:
        print ("ERR: Not receiving anything. Something's wrong with your setup.")
        exit_clean()

    
    tx_bram_dmp=dict()
    for bram in brams:
        bram_name = tx_snap+'_'+bram
        print 'Reading %i values from bram %s...'%(tx_size,bram_name),
        tx_bram_dmp[bram]=fpga.read(bram_name,tx_size*4)
        sys.stdout.flush()
        print 'ok'

    rx_bram_dmp=dict()
    for bram in brams:
        bram_name = rx_snap+'_'+bram
        print 'Reading %i values from bram %s...'%(rx_size,bram_name),
        rx_bram_dmp[bram]=fpga.read(bram_name,rx_size*4)
        sys.stdout.flush()
        print 'ok'

    print 'Unpacking TX packet stream...'
    tx_data=[]
    ip_mask = (2**(24+5)) -(2**5) # beacuse bram_oob is 32bits, and some bits are required for other information, we can only reliably get the 3 lower bytes of the IP address
    for i in range(0,tx_size):
        data_64bit = struct.unpack('>Q',tx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+tx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
        tx_data.append(data_64bit)
        if not args.silent:
            oob_32bit = struct.unpack('>L',tx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
            print '[%4i]: data: 0x%016X'%(i,data_64bit),
            ip_string = socket.inet_ntoa(struct.pack('>L',(oob_32bit&(ip_mask))>>5))            
            print 'Dest IP: %s'%(ip_string),
            if oob_32bit&(2**0): print '[TX overflow]',
            if oob_32bit&(2**1): print '[TX almost full]',
            if oob_32bit&(2**2): print '[tx_active]',
            if oob_32bit&(2**3): print '[link_up]',
            if oob_32bit&(2**4): print '[eof]',
            print '' 

    print 'Unpacking RX packet stream...'
    rx_data=[]
    ip_mask = (2**(24+5)) -(2**5) #24 bits, starting at bit 5 are valid for ip address (from snap block)
    for i in range(0,rx_size):
        data_64bit = struct.unpack('>Q',rx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+rx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
        rx_data.append(data_64bit)
        if not args.silent:
            oob_32bit = struct.unpack('>L',rx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
            print '[%4i]: data: 0x%016X'%(i,data_64bit),
            ip_string = socket.inet_ntoa(struct.pack('>L',(oob_32bit&(ip_mask))>>5))
            print 'IP: %s'%(ip_string),
            if oob_32bit&(2**0): print '[RX overrun]',
            if oob_32bit&(2**1): print '[RX bad frame]',
            if oob_32bit&(2**3): print '[rx_active]',
            if oob_32bit&(2**4): print '[link_up]',
            if oob_32bit&(2**2): print '[eof]',
            print '' 

    print 'Checking data TX vs data RX...',
    okay = True
    for i in range(0, len(tx_data)):
        try:
            assert(tx_data[i] == rx_data[i])
        except AssertionError:
            print 'TX[%i](%i) != RX[%i](%i)' % (i, tx_data[i], i, rx_data[i])
            okay = False
    if okay:
        print 'ok.'
    else:
        print 'ERROR.'

    print '=========================='

    if args.plot:   
        import pylab
        pylab.subplot(211)
        pylab.plot(tx_data, label='TX data')
        pylab.subplot(212)
        pylab.plot(rx_data, label='RX data')
        pylab.show()

except KeyboardInterrupt:
    exit_clean()
except Exception as inst:
    print type(inst)
    print inst.args
    print inst
    exit_fail()

exit_clean()
