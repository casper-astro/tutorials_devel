#!/usr/bin/python

# tut5.py
# CASPER Tutorial 5: Heterogeneous Instrumentation
#   Config script.

import corr
import time
import numpy
import math
import struct
import sys

bitstream = 'tut5_2012_Aug_06_2116.bof'
katcpPort = 7147
roach = '192.168.40.70'

destIP = 10*(2**24) + 1*(2**8) + 146    # 10.0.1.146
destPort = 60000
srcIP = 10*(2**24) + 145                # 10.0.0.145
srcPort = 50000
MACBase = (2<<40) + (2<<32)
gbe0 = 'gbe0'

def writeData(Tone1Freq, Tone2Freq, SampFreq):
    ScaleFactor = 127
    TimeSamples = 256   # time samples per packet
    realPart = numpy.array([ScaleFactor * 0.1 * math.cos(2 * math.pi * Tone1Freq * i / SampFreq) for i in range(TimeSamples)])
    realPart = realPart +  numpy.array([ScaleFactor * 0.2 * math.cos(2 * math.pi * Tone2Freq * i / SampFreq) for i in range(TimeSamples)])
    realPart = realPart + numpy.random.random_integers(-64, 64, TimeSamples)
    realPart = realPart.astype(numpy.int8)
    imagPart = numpy.array([ScaleFactor * 0.1 * math.sin(2 * math.pi * Tone1Freq * i / SampFreq) for i in range(TimeSamples)])
    imagPart = imagPart + numpy.array([ScaleFactor * 0.2 * math.sin(2 * math.pi * Tone2Freq * i / SampFreq) for i in range(TimeSamples)])
    imagPart = imagPart + numpy.random.random_integers(-64, 64, TimeSamples)
    imagPart = imagPart.astype(numpy.int8)
    interleavedData = numpy.empty(TimeSamples * 4, dtype=numpy.int8)
    interleavedData[0::4] = realPart    # x-pol real
    interleavedData[1::4] = imagPart    # x-pol imag
    interleavedData[2::4] = realPart    # y-pol real
    interleavedData[3::4] = imagPart    # y-pol real
    MSB = numpy.reshape(interleavedData, [TimeSamples, 4])[0::2]
    LSB = numpy.reshape(interleavedData, [TimeSamples, 4])[1::2]
    MSB = numpy.reshape(MSB, [1, TimeSamples * 2])
    LSB = numpy.reshape(LSB, [1, TimeSamples * 2])
    devName = 'pkt_sim_bram_msb'
    fpga.write(devName, struct.pack('>512b', *MSB[0]))
    devName = 'pkt_sim_bram_lsb'
    fpga.write(devName, struct.pack('>512b', *LSB[0]))

print 'Connecting to server %s... ' %(roach),
sys.stdout.flush()
fpga = corr.katcp_wrapper.FpgaClient(roach, katcpPort)
time.sleep(1)
if fpga.is_connected():
    print 'DONE'
else:
    print 'ERROR: Failed to connect to server %s!' %(roach)
    sys.exit(0);

print 'Programming FPGA with %s...' %bitstream,
sys.stdout.flush()
fpga.progdev(bitstream)
print 'DONE'

time.sleep(5)

fpga.listdev()
gbe0_link = bool(fpga.read_int(gbe0))
if not gbe0_link:
   print 'ERROR: There is no cable plugged into port0!'

#print 'Starting 10GbE driver...',   
#sys.stdout.flush()
#fpga.tap_start('tap0', gbe0, MACBase + srcIP, srcIP, srcPort)
#print 'DONE'

print 'Setting up packet generation...',
sys.stdout.flush()
fpga.write_int('dest_ip', destIP)
fpga.write_int('dest_port', destPort)

# send data at a rate
#   = (pkt_sim_payload_len / pkt_sim_period) * 10GbE core clock rate
fpga.write_int('pkt_sim_period', 512)
fpga.write_int('pkt_sim_payload_len', 128)  # send 128 * 8 bytes of data
                                            # per packet
print 'DONE'

print 'Enabling the packetizer...',
sys.stdout.flush()
fpga.write_int('pkt_sim_enable', 1)
print 'DONE'

print 'Writing test data to BRAMs...',
sys.stdout.flush()
SampFreq = 32e6     # 32 MHz
Tone1Freq = 10e6    # 10 MHz
Tone2Freq = 16e6    # 16 MHz
writeData(Tone1Freq, Tone2Freq, SampFreq)
print 'DONE'

