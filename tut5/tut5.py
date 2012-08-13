#!/usr/bin/env python

# tut5.py
# CASPER Tutorial 5: Heterogeneous Instrumentation
#   Config script.

import corr
import time
import numpy
import math
import struct
import sys

katcpPort = 7147
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

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut5.py <ROACH_HOSTNAME_or_IP> <GPU server 10GbE IP>')
    p.set_description(__doc__)

    p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=(2**28)/1024,
        help='Set the number of vectors to accumulate between dumps. default is (2^28)/1024.')
    p.add_option('-g', '--gain', dest='gain', type='int',default=1000,
        help='Set the digital gain (4bit quantisation scalar). default is 1000.')
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-b', '--bof', dest='boffile', type='str', default='',
        help='Specify the bof file to load')
    opts, args = p.parse_args(sys.argv[1:])

    if len(args)!=2:
        print 'Please specify a ROACH board and GPU server 10GbE IP. \nExiting.'
        exit()
    else:
        roach = args[0]
        destIP = args[1]
        tmp = destIP.split('.')
        destIP = int(tmp[0])*(2**24) + int(tmp[1])*(2**16) + int(tmp[2])*(2**8) + int(tmp[3])
        print destIP

    if opts.boffile != '':
        boffile = opts.boffile
    else:
        boffile = 'tut5.bof'


    print 'Connecting to server %s... ' %(roach),
    sys.stdout.flush()
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcpPort)
    time.sleep(1)
    if fpga.is_connected():
        print 'DONE'
    else:
        print 'ERROR: Failed to connect to server %s!' %(roach)
        sys.exit(0);

    print 'Programming FPGA with %s...' %boffile,
    sys.stdout.flush()
    fpga.progdev(boffile)
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

