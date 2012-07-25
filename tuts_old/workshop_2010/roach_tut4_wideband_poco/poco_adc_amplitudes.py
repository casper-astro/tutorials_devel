#!/usr/bin/env python

'''
Reads the values of the RMS amplitude accumulators on the ROACH.
Requires PoCo rev 314 or later.
'''
import corr, time, numpy, struct, sys, logging

pol_map=['I','Q']
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
    p.set_usage('poco_adc_amplitudes.py ROACH')
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. \nExiting.'
        exit()
    else:
        roach = args[0]

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

    while(1):
        #get the data...    
        inputs=[{},{},{},{}]
        for input in range(4):
            regname = 'adc_sum_sq%i'%(input)
            inputs[input]['unpacked'] = fpga.read_uint(regname)
            inputs[input]['rms'] = numpy.sqrt(inputs[input]['unpacked']/(2.0**16))/(2**(8-1))
            if inputs[input]['unpacked'] == 0:
                inputs[input]['bits_used'] = 0        
            else:    
                inputs[input]['bits_used'] = (numpy.log2(inputs[input]['rms'] * (2**8)))

        #print the output                
        time.sleep(1)
        #Move cursor home:
        print '%c[H'%chr(27)
        #clear the screen:
        print '%c[2J'%chr(27)
        print ' ADC amplitudes'
        print '----------------'
        for a in range(2):
            for p in range(2):
                print 'ADC%i input %s: %.3f (%2.2f bits used)'%(a,pol_map[p],inputs[a*2+p]['rms'],inputs[a*2+p]['bits_used'])
        print '------------------------------------'

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()
 
