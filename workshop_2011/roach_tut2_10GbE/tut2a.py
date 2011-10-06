#!/usr/bin/env python

import corr,time,numpy,struct,sys

bitstream = 'tut2a.bof'
roach     = 'ROACH03'

#Decide where we're going to send the data, and from which addresses:
dest_ip  = 10*(2**24) + 145 #10.0.0.145
port     = 46224
src_ip   = 10*(2**24) + 10  #10.0.0.10
mac_base = (2<<40) + (2<<32)
gbe0     = 'gbe0';

print('Connecting to server %s on port... '%(roach)),
fpga = corr.katcp_wrapper.FpgaClient(roach)
time.sleep(2)

if fpga.is_connected():
	print 'ok\n'	
else:
    print 'ERROR\n'

print '------------------------'
print 'Programming FPGA with %s...' %bitstream,
fpga.progdev(bitstream)
print 'ok\n'
time.sleep(5)

print '------------------------'
print 'Setting the port 0 linkup :',
fpga.listdev()
gbe0_link=bool(fpga.read_int(gbe0))
print gbe0_link
if not gbe0_link:
   print 'There is no cable plugged into port0'
print '------------------------'
print 'Configuring receiver core...',   
fpga.tap_start('tap0',gbe0,mac_base+src_ip,src_ip,port)
print 'done'

print '------------------------'
print 'Setting-up packet core...',
sys.stdout.flush()
fpga.write_int('dest_ip',dest_ip)
fpga.write_int('dest_port',port)
fpga.write_int('pkt_sim_period',200)
fpga.write_int('pkt_sim_length',6)
print 'done'

print 'Enabling the packetizer...',
fpga.write_int('pkt_sim_enable',0)
print 'done'

