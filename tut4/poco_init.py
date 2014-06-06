#!/usr/bin/env python
'''
This script demonstrates programming an FPGA and configuring a wideband Pocket correlator using the Python KATCP library along with the katcp_wrapper distributed in the corr package. Designed for use with CASPER workshop Tutorial 4.
\n\n 
Author: Jason Manley, August 2010.
Modified: May 2012, Medicina.
Modified: Aug 2012, Nie Jun
'''

#TODO: add support for coarse delay change
#TODO: add support for ADC histogram plotting.
#TODO: add support for determining ADC input level 

import corr,time,numpy,struct,sys,logging,pylab

katcp_port=7147

def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    exit()

def exit_clean():
    try:
        fpga.stop()
    except: pass
    exit()


if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('poco_init.py <ROACH_HOSTNAME_or_IP> [options]')
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

    if args==[]:
        print 'Please specify a ROACH board. \nExiting.'
        exit()
    else:
        roach = args[0]

    if opts.boffile != '':
        boffile = opts.boffile
    else:
        boffile = 'tut4.bof'

try:
    loggers = []
    lh=corr.log_handlers.DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s on port %i... '%(roach,katcp_port)),
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10,logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(roach,katcp_port)
        exit_fail()

    print '------------------------'
    print 'Programming FPGA...',
    if not opts.skip:
        fpga.progdev(boffile)
        print 'done'
    else:
        print 'Skipped.'

    print 'Configuring fft_shift...',
    fpga.write_int('fft_shift',(2**32)-1)
    print 'done'

    print 'Configuring accumulation period...',
    fpga.write_int('acc_len',opts.acc_len)
    print 'done'

    print 'Resetting board, software triggering and resetting error counters...',
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<17) #arm
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<18) #software trigger
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<18) #issue a second trigger
    print 'done'

    #EQ SCALING!
    # writes only occur when the addr line changes value. 
    # write blindly - don't bother checking if write was successful. Trust in TCP!
    print 'Setting gains of all channels on all inputs to %i...'%opts.gain,
    fpga.write_int('quant0_gain',opts.gain) #write the same gain for all inputs, all channels
    fpga.write_int('quant1_gain',opts.gain) #write the same gain for all inputs, all channels
    fpga.write_int('quant2_gain',opts.gain) #write the same gain for all inputs, all channels
    fpga.write_int('quant3_gain',opts.gain) #write the same gain for all inputs, all channels
    for chan in range(1024):
        #print '%i...'%chan,
        sys.stdout.flush()
        for input in range(4):
            fpga.blindwrite('quant%i_addr'%input,struct.pack('>I',chan))
    print 'done'

    print "ok, all set up. Try use poco_adc_amplitudes.py to determine the adc input level, then \
           try plotting using poco_plot_auto.py or poco_plot_cross.py"

#    time.sleep(2)
#
#   prev_integration = fpga.read_uint('acc_num')
#   while(1):
#       current_integration = fpga.read_uint('acc_num')
#       diff=current_integration - prev_integration
#       if diff==0:
#           time.sleep(0.01)
#       else:
#           if diff > 1:
#               print 'WARN: We lost %i integrations!'%(current_integration - prev_integration)
#           prev_integration = fpga.read_uint('acc_num')
#           print 'Grabbing integration number %i'%prev_integration
#           
#           if opts.auto:
#               plot_autos()
#           else:
#               plot_cross(opts.cross)
#
except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

