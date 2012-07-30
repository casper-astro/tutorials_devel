#!/usr/bin/python2.6

import corr, time, numpy, math, struct, sys, pylab

bitstream = 'tut5_2012_Jul_30_1325.bof'
katcp_port = 7147
roach = '192.168.40.70'

dest_ip = 10*(2**24) + 145    #10.0.0.145
dest_port = 60000
src_ip = 10*(2**24) + 10      #10.0.0.10
src_port = 50000
mac_base = (2<<40) + (2<<32)
gbe0 = 'gbe0'

def writeData(ToneFreq, SampFreq):
    ScaleFactor = 127
    TimeSamples = 256   # time samples per packet
    realPart = numpy.array([ScaleFactor * 0.1 * math.cos(2 * math.pi * ToneFreq * i / SampFreq) for i in range(TimeSamples)])
    realPart = realPart.astype(numpy.int8)
    imagPart = numpy.array([ScaleFactor * 0.1 * math.sin(2 * math.pi * ToneFreq * i / SampFreq) for i in range(TimeSamples)])
    imagPart = imagPart.astype(numpy.int8)
    interleavedData = numpy.empty(TimeSamples * 4, dtype=numpy.int8)
    interleavedData[0::4] = realPart    # x-pol real
    interleavedData[1::4] = imagPart    # x-pol imag
    interleavedData[2::4] = realPart    # y-pol real
    interleavedData[3::4] = imagPart    # y-pol real
    MSB = numpy.reshape(interleavedData, [4, TimeSamples])[0::2]
    LSB = numpy.reshape(interleavedData, [4, TimeSamples])[1::2]
    MSB = numpy.reshape(MSB, [1, TimeSamples * 2])
    LSB = numpy.reshape(LSB, [1, TimeSamples * 2])
    devName = 'pkt_sim_bram_msb'
    fpga.write(devName, struct.pack('>512b', *MSB[0]))
    devName = 'pkt_sim_bram_lsb'
    fpga.write(devName, struct.pack('>512b', *LSB[0]))
    #pylab.plot(realPart)
    #pylab.show()

print 'Connecting to server %s... ' %(roach),
sys.stdout.flush()
fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port)
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
print gbe0_link
if not gbe0_link:
   print 'ERROR: There is no cable plugged into port0!'

#print 'Starting 10GbE driver...',   
#sys.stdout.flush()
#fpga.tap_start('tap0', gbe0, mac_base + src_ip, src_ip, src_port)
#print 'DONE'

print 'Setting up packet generation...',
sys.stdout.flush()
fpga.write_int('dest_ip', dest_ip)
fpga.write_int('dest_port', dest_port)

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
SampFreq = 128e6    # 128 MHz
ToneFreq = 5e6      # 1 MHz
writeData(ToneFreq, SampFreq)
print 'DONE'

