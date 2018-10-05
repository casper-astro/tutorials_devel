#!/usr/bin/env python
'''
This code demonstrates readount of a SNAP spectrometer. You need a SNAP with:
-- A 10 MHz, 8dBm  reference going into the SYNTH_OSC SMA (3rd SMA from the left)
-- a test tone going into the ADC0 input 0 (8th input from the left)
'''

#TODO: add support for ADC histogram plotting.
#TODO: add support for determining ADC input level 

import casperfpga,casperfpga.snapadc,time,numpy,struct,sys,logging,pylab,matplotlib

katcp_port=7147


#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser


    p = OptionParser()
    p.set_usage('spectrometer.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='snap_tut_corr.fpg',
        help='Specify the fpg file to load')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a SNAP board. Run with the -h flag to see all options.\nExiting.'
        exit()
    else:
        snap = args[0] 
    if opts.fpgfile != '':
        bitstream = opts.fpgfile

try:

    print('Connecting to server %s on port %i... '%(snap,katcp_port)),
    fpga = casperfpga.CasperFpga(snap)
    time.sleep(0.2)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(snap,katcp_port)
        exit_fail()

    print '------------------------'
    print 'Programming FPGA with %s...' %bitstream,
    sys.stdout.flush()
    if not opts.skip:
        fpga.upload_to_ram_and_program(bitstream)
        print 'done'
    else:
        print 'Skipped.'

    # After programming we need to configure the ADC. The following function call assumes that
    # the SNAP has a 10 MHz reference connected. It will use this reference to generate an 800 MHz
    # sampling clock. The init function will also tweak the alignment of the digital lanes that
    # carry data from the ADC chips to the FPGA, to ensure reliable data capture. It should take about
    # 30 seconds to run.
    # In the future you won't have to instantiate an adc object as below, casperfpga will automatically
    # detect the presence of an adc block in your SNAP design, and will automagically create you
    # an adc object to interact with.
    adc = casperfpga.snapadc.SNAPADC(fpga, ref=10) # reference at 10MHz
    # We want a sample rate of 800 Mhz, with 1 channel per ADC chip, using 8-bit ADCs
    # (there is another version of the ADC chip which operates with 12 bits of precision)
    if not opts.skip:
        print 'Attempting to initialize ADC chips...'
        sys.stdout.flush()
        # try initializing a few times for good measure in case it fails...
        done = False
        for i in range(3):
            if adc.init(samplingRate=800, numChannel=1, resolution=8) == 0:
                done = True
                break
        print 'done (took %d attempts)' % (i+1)
        if not done:
            print 'Failed to calibrate after %d attempts' % (i+1)
            exit_clean()
        
    # Since we're in 4-way interleaving mode (i.e., one input per snap chip) we should configure
    # the ADC inputs accordingly
    adc.selectADC([0,1,2]) # send commands to the all three ADC chips
    adc.adc.selectInput([1,1,1,1]) # Interleave four ADCs all pointing to the first input

    print 'Configuring accumulation period...',
    sys.stdout.flush()
    fpga.write_int('acc_len',opts.acc_len)
    print 'done'


except KeyboardInterrupt:
    exit()

exit()

