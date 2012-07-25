#!/usr/bin/env python

'''
This script demonstrates programming an FPGA, configuring 10GbE cores and checking transmitted and received data using the Python KATCP library along with the katcp_wrapper distributed in the corr package. Designed for use with TUT3 at the 2009 CASPER workshop.
\n\n 
Author: Jason Manley, August 2009.
'''
import corr, time, struct, sys, logging, socket

#Decide where we're going to send the data, and from which addresses:
dest_ip  =10*(2**24) + 30 #10.0.0.30
fabric_port=60000         
source_ip=10*(2**24) + 20 #10.0.0.20
mac_base=(2<<40) + (2<<32)
ip_prefix='10. 0. 0.'     #Used for the purposes of printing output because snap blocks only store least significant word.

pkt_period = 16384  #in FPGA clocks (200MHz)
payload_len = 128   #in 64bit words

brams=['bram_msb','bram_lsb','bram_oob']
tx_snap = 'snap_gbe0_tx'
rx_snap = 'snap_gbe3_rx'

tx_core_name = 'gbe0'
rx_core_name = 'gbe3'

#boffile='tut2_2009_Sep_28_1407.bof'
boffile = 'tut2_2010_Apr_02_1057.bof'
fpga=[]

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
#    try:
#        fpga.stop()
#    except: pass
#    raise
    exit()

def exit_clean():
    try:
        for f in fpgas: f.stop()
    except: pass
    exit()

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut2.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-p', '--plot', dest='plot', action='store_true',
        help='Plot the TX and RX counters. Needs matplotlib/pylab.')  
    p.add_option('-a', '--arp', dest='arp', action='store_true',
        help='Print the ARP table and other interesting bits.')  
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. \nExiting.'
        exit()
    else:
        roach = args[0]

try:
    lh=corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(roach)),
    fpga = corr.katcp_wrapper.FpgaClient(roach, logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(roach)
        exit_fail()
    
    print '------------------------'
    print 'Programming FPGA...',
    sys.stdout.flush()
    fpga.progdev(boffile)
    time.sleep(10)
    print 'ok'

    print '---------------------------'
    print 'Port 0 linkup: ',
    sys.stdout.flush()
    gbe0_link=bool(fpga.read_int('gbe0_linkup'))
    print gbe0_link
    if not gbe0_link:
        print 'There is no cable plugged into port0. Please plug a cable into ports 0 and 3 to continue demo. Exiting.'
        exit_clean()
    print 'Port 3 linkup: ',
    sys.stdout.flush()
    gbe3_link=bool(fpga.read_int('gbe3_linkup'))
    print gbe3_link
    if not gbe0_link:
        print 'There is no cable plugged into port3. Please plug a cable into ports 0 and 3 to continue demo. Exiting.'
        exit_clean()

    print '---------------------------'
    print 'Configuring receiver core...',
    sys.stdout.flush()
    fpga.tap_start('tap0',rx_core_name,mac_base+dest_ip,dest_ip,fabric_port)
    print 'done'
    print 'Configuring transmitter core...',
    sys.stdout.flush()
    fpga.tap_start('tap3',tx_core_name,mac_base+source_ip,source_ip,fabric_port)
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
    fpga.write_int('rst',3)
    fpga.write_int('rst',0)
    print 'done'

    time.sleep(2)

    if opts.arp:
        print '\n\n==============================='
        print '10GbE Transmitter core details:'
        print '==============================='
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        fpga.print_10gbe_core_details(tx_core_name,arp=True)
        print '\n\n============================'
        print '10GbE Receiver core details:'
        print '============================'
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        fpga.print_10gbe_core_details(rx_core_name,arp=True)

    print 'Sent %i packets already.'%fpga.read_int('gbe0_tx_cnt')
    print 'Received %i packets already.'%fpga.read_int('gbe3_rx_frame_cnt')

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
    fpga.write_int('pkt_sim_enable',1)
    print 'done'

    time.sleep(2)

    tx_size=fpga.read_int(tx_snap+'_addr')+1
    rx_size=fpga.read_int(rx_snap+'_addr')+1
    if tx_size <= 1:
        print ('ERR: Not transmitting anything. This should not happen. Exiting.')
        exit_clean()
    if rx_size <= 1:
        print ('ERR: Not receiving anything.')

    
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
    for i in range(0,tx_size):
        data_64bit = struct.unpack('>Q',tx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+tx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
        oob_32bit = struct.unpack('>L',tx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
        print '[%4i]: data: %16X'%(i,data_64bit),
        ip_mask = (2**(8+5)) -(2**5)
        print 'IP: %s%3d'%(ip_prefix,(oob_32bit&(ip_mask))>>5),
        if oob_32bit&(2**0): print '[TX overflow]',
        if oob_32bit&(2**1): print '[TX almost full]',
        if oob_32bit&(2**2): print '[TX LED]',
        if oob_32bit&(2**3): print '[Link up]',
        if oob_32bit&(2**4): print '[eof]',
        tx_data.append(data_64bit)
        print '' 

    print 'Unpacking RX packet stream...'
    rx_data=[]
    for i in range(0,rx_size):
        data_64bit = struct.unpack('>Q',rx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+rx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
        oob_32bit = struct.unpack('>L',rx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
        print '[%4i]: data: %16X'%(i,data_64bit),
        ip_mask = (2**(24+5)) -(2**5)
        ip_string = socket.inet_ntoa(struct.pack('>L',(oob_32bit&(ip_mask))>>5))
        print 'IP: %s'%(ip_string),
        if oob_32bit&(2**0): print '[RX overrun]',
        if oob_32bit&(2**1): print '[RX bad frame]',
        if oob_32bit&(2**3): print '[led_rx]',
        if oob_32bit&(2**4): print '[led_up]',
        if oob_32bit&(2**2): print '[eof]',
        rx_data.append(data_64bit)
        print '' 

    print '=========================='

    if opts.plot:
        import pylab
        pylab.subplot(211)
        pylab.plot(tx_data,label='TX data')
        pylab.subplot(212)
        pylab.plot(rx_data,label='RX data')
        pylab.show()

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

