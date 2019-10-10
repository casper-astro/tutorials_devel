#!/bin/env ipython

'''
This script demonstrates:
- programming an FPGA
- feeding the adder with inputs and checking back the results
- controlling and monitoring the counter.

Designed for use with tut_intro at the 2019 CASPER workshop.
\n\n 
Author: Cedric Viou, 2019.
'''
import casperfpga, time, struct, sys, logging, socket, numpy


fpgfile = 'snap_tut_intro.fpg'


def exit_fail():
    logger.fatal('FAILURE DETECTED.' )
    sys.exit()

def exit_clean():
    sys.exit()

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('snap_tut_intro.py <CASPER_FPGA_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('', '--noprogram', dest='noprogram', action='store_true',
        help='Don\'t programm FPGA')  
    p.add_option('-b', '--fpgfile', dest='fpg', type='str', default=fpgfile,
        help='Specify the fpg file to load')  
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print('Please specify a CASPER FPGA board. \nExiting.')
        sys.exit()
    else:
        casper_fpga = args[0]
    if opts.fpg != '':
        fpgfile = opts.fpg
try:

    logger = logging.getLogger(casper_fpga)
    
    fpga = casperfpga.CasperFpga(casper_fpga, logger=logger, log_level='INFO')
    time.sleep(1)


    if fpga.is_connected():
        fpga.logger.info('FPGA connected')
    else:
        fpga.logger.error('ERROR connecting to server %s.\n'%(casper_fpga))
        exit_fail()
    
    
    
    if not opts.noprogram:
        fpga.logger.info('Programming FPGA at %s...' % casper_fpga)
        fpga.upload_to_ram_and_program(fpgfile)
        time.sleep(10)
        fpga.logger.info('ok')



    fpga.logger.info('Go see on the board if LED0 is blinking')
    time.sleep(1)



    fpga.logger.info("Let's do some math...")
    Nb_test = 100
    for a, b in numpy.random.randint(0, 2**31, size=(Nb_test, 2)):
        fpga.write_int('a', a, blindwrite=True,)
        fpga.write_int('b', b, blindwrite=True,)
        sum_a_b = fpga.read_uint('sum_a_b')
        assert a + b == sum_a_b, "%u + %u != %u -> check that..." % (a, b, sum_a_b)
        fpga.logger.info("  %10u + %10u = %10u\r" % (a, b, sum_a_b))
    fpga.logger.info("Done adding %u pairs of uint: no errors" % Nb_test)



    fpga.logger.info("Let's look at the counter...")
    for idx in range(3):
        fpga.logger.info("    %u" % fpga.read_uint('counter_value'))
        
    fpga.logger.info("Enable counter")
    fpga.write_int('counter_ctrl', 1)
    for idx in range(30):
        fpga.logger.info("    %u" % fpga.read_uint('counter_value'))
        time.sleep(0.1)
        
    fpga.logger.info("Reset counter")
    fpga.write_int('counter_ctrl', 3)
    fpga.write_int('counter_ctrl', 1)
    for idx in range(30):
        fpga.logger.info("    %u" % fpga.read_uint('counter_value'))
        time.sleep(0.1)
        
    fpga.logger.info("Stop counter")
    fpga.write_int('counter_ctrl', 0)
    for idx in range(10):
        fpga.logger.info("    %u" % fpga.read_uint('counter_value'))
        time.sleep(0.1)
        
    fpga.logger.info("Reset counter")
    fpga.write_int('counter_ctrl', 2)
    fpga.write_int('counter_ctrl', 0)
    for idx in range(10):
        fpga.logger.info("    %u" % fpga.read_uint('counter_value'))
        time.sleep(0.1)




except KeyboardInterrupt:
    exit_clean()
except Exception as inst:
    print type(inst)
    print inst.args
    print inst
    exit_fail()

exit_clean()
