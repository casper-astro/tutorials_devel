#!/bin/env ipython

'''
This script demonstrates programming an FPGA, configuring 10GbE cores and checking transmitted and received data using the Python KATCP library along with the katcp_wrapper distributed in the corr package. Designed for use with TUT3 at the 2009 CASPER workshop.
\n\n 
Author: Jason Manley, August 2009.
Updated for CASPER 2013 workshop. This tut needs a rework to use new snap blocks and auto bit unpack.
'''
import casperfpga, time, struct, sys, logging, socket, numpy

#Decide where we're going to send the data, and from which addresses:
fabric_port= 60000         
mac_base= (2<<40) + (2<<32)
ip_base = 192*(2**24) + 168*(2**16) + 10*(2**8)

pkt_period = 16384  #how often to send another packet in FPGA clocks (200MHz)
payload_len = 128   #how big to make each packet in 64bit words

tx_snap = 'tx_snapshot'

tx_core_name = 'gbe0'
rx_core_name = 'gbe1'

fpgfile = 'dgorthi_tut2_snap.fpg'
fpga=[]

def exit_fail():
#    print 'FAILURE DETECTED. Log entries:\n'#,lh.printMessages()
#    try:
#        fpga.stop()
#    except: pass
#    raise
    sys.exit()

def exit_clean():
    try:
        for f in fpgas: f.stop()
    except: pass
    sys.exit()

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut2.py <SNAP_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('', '--noprogram', dest='noprogram', action='store_true',
        help='Don\'t print the contents of the packets.')  
    p.add_option('-s', '--silent', dest='silent', action='store_true',
        help='Don\'t print the contents of the packets.')  
    p.add_option('-p', '--plot', dest='plot', action='store_true',
        help='Plot the TX and RX counters. Needs matplotlib/pylab.')  
    p.add_option('-a', '--arp', dest='arp', action='store_true',
        help='Print the ARP table and other interesting bits.')  
    p.add_option('-b', '--fpgfile', dest='fpg', type='str', default=fpgfile,
        help='Specify the fpg file to load')  
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a SNAP board. \nExiting.'
        sys.exit()
    else:
        snap = args[0]
    if opts.fpg != '':
        fpgfile = opts.fpg
try:
    #lh = corr.log_handlers.DebugLogHandler()
    lh = logging.StreamHandler()
    #lh = casperfpga.casperfpga.logging.getLogger()
    logger = logging.getLogger(snap)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(snap)),
    fpga = casperfpga.CasperFpga(snap,logger=logger)
    #fpga = corr.katcp_wrapper.FpgaClient(snap, logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(snap)
        exit_fail()
    
    if not opts.noprogram:
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
        #exit_clean()
    print 'Port 1 linkup: ',
    sys.stdout.flush()
    gbe1_link=bool(fpga.read_int('gbe1_linkup'))
    print gbe1_link
    if not gbe1_link:
        print 'There is no cable plugged into port1. Please plug a cable between ports 0 and 1 to continue demo. Exiting.'
        #exit_clean()

    print '---------------------------'
    print 'Configuring receiver core...',
    sys.stdout.flush()
    gbe_rx = fpga.gbes[rx_core_name]
    gbe_rx.set_arp_table(mac_base+numpy.arange(256))
    gbe_rx.setup(mac_base+13,ip_base+13,fabric_port)
    #fpga.tap_start('tap0',rx_core_name,mac_base+dest_ip,dest_ip,fabric_port)
    print 'done'

    print 'Configuring transmitter core...',
    sys.stdout.flush()
    #fpga.tap_start('tap3',tx_core_name,mac_base+source_ip,source_ip,fabric_port)
    #Set IP address of snap and set arp-table
    gbe_tx = fpga.gbes[tx_core_name]
    gbe_tx.set_arp_table(mac_base+numpy.arange(256))
    gbe_tx.setup(mac_base+16,ip_base+16,fabric_port)
    print 'done'

    print 'Setting-up destination addresses...',
    sys.stdout.flush()
    fpga.write_int('dest_ip',ip_base+13)
    fpga.write_int('dest_port',fabric_port)
    print 'done'

    print '---------------------------'
    print 'Setting-up packet source...',
    sys.stdout.flush()
    fpga.write_int('pkt_sim_period',pkt_period)
    fpga.write_int('pkt_sim_payload_len',payload_len)
    print 'done'

    print 'Resetting cores and counters...',
    sys.stdout.flush()
    fpga.write_int('rst', 3)
    fpga.write_int('rst', 0)
    print 'done'

    time.sleep(2)

    if opts.arp:
        print '\n\n==============================='
        print '10GbE Transmitter core details:'
        print '==============================='
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        tx_gbe.print_10gbe_core_details()
        print '\n\n============================'
        print '10GbE Receiver core details:'
        print '============================'
        print "Note that for some IP address values, only the lower 8 bits are valid!"
        gbe_rx.print_10gbe_core_details()

    print 'Sent %i packets already.'%fpga.read_int('gbe0_tx_cnt')
    print 'Received %i packets already.'%fpga.read_int('gbe1_rx_frame_cnt')

    print 'Enabling output...',
    sys.stdout.flush()
    fpga.write_int('pkt_sim_enable', 1)
    print 'done'

    time.sleep(2)

    print '------------------------'
    print 'Triggering snap captures...',
    sys.stdout.flush()

    txss = fpga.snapshots['tx_snapshot_ss']
    print 'Reading %i values from bram %s...'%(2*payload_len,'tx_snapshot'),
    sys.stdout.flush()
    txss.print_snap(2*payload_len,man_valid=False,man_trig=False,timeout=10)
    print 'done'

    print '-------------------------'
    print 'Reading TX meta data'
    gbe_tx_ss = gbe_tx.read_txsnap()
    
    if sum(gbe_tx_ss['tx_full']):
        print 'The TX core is overflowing!!'
    
    print '-------------------------'
    print 'Reading RX meta data'        
    gbe_rx_ss = gbe_rx.read_rxsnap()

    if sum(gbe_tx_ss['tx_full']):
        print 'The RX core is overflowing!!'
    
# Count the pkts rcved and transmed
# Rate check of tx and rx - print, check
# Snap pkts- Compare rx and tx data
    
    # rx_bram_dmp=dict()
    # for bram in brams:
    #     bram_name = rx_snap+'_'+bram
    #     print 'Reading %i values from bram %s...'%(rx_size,bram_name),
    #     rx_bram_dmp[bram]=fpga.read(bram_name,rx_size*4)
    #     sys.stdout.flush()
    #     print 'ok'

    # print 'Unpacking TX packet stream...'
    # tx_data=[]
    # for i in range(0,tx_size):
    #     data_64bit = struct.unpack('>Q',tx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+tx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
    #     tx_data.append(data_64bit)
    #     if not opts.silent:
    #         oob_32bit = struct.unpack('>L',tx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
    #         print '[%4i]: data: 0x%016X'%(i,data_64bit),
    #         ip_mask = (2**(8+5)) -(2**5)
    #         print 'IP: 0.0.0.%03d'%((oob_32bit&(ip_mask))>>5),
    #         if oob_32bit&(2**0): print '[TX overflow]',
    #         if oob_32bit&(2**1): print '[TX almost full]',
    #         if oob_32bit&(2**2): print '[tx_active]',
    #         if oob_32bit&(2**3): print '[link_up]',
    #         if oob_32bit&(2**4): print '[eof]',
    #         print '' 

    # print 'Unpacking RX packet stream...'
    # rx_data=[]
    # ip_mask = (2**(24+5)) -(2**5) #24 bits, starting at bit 5 are valid for ip address (from snap block)
    # for i in range(0,rx_size):
    #     data_64bit = struct.unpack('>Q',rx_bram_dmp['bram_msb'][(4*i):(4*i)+4]+rx_bram_dmp['bram_lsb'][(4*i):(4*i)+4])[0]
    #     rx_data.append(data_64bit)
    #     if not opts.silent:
    #         oob_32bit = struct.unpack('>L',rx_bram_dmp['bram_oob'][(4*i):(4*i)+4])[0]
    #         print '[%4i]: data: 0x%016X'%(i,data_64bit),
    #         ip_string = socket.inet_ntoa(struct.pack('>L',(oob_32bit&(ip_mask))>>5))
    #         print 'IP: %s'%(ip_string),
    #         if oob_32bit&(2**0): print '[RX overrun]',
    #         if oob_32bit&(2**1): print '[RX bad frame]',
    #         if oob_32bit&(2**3): print '[rx_active]',
    #         if oob_32bit&(2**4): print '[link_up]',
    #         if oob_32bit&(2**2): print '[eof]',
    #         print '' 

    print 'Checking data TX vs data RX...',
    tx_data = gbe_tx_ss['data']
    rx_data = gbe_rx_ss['data']
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
    if opts.plot:   
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

