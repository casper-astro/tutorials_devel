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
import casperfpga, time, sys, numpy, os.path

fpgfile = 'roach2_tut_intro.fpg'


def exit_fail():
    fpga.logger.fatal('FAILURE DETECTED.' )
    sys.exit()

def exit_clean():
    sys.exit()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
             description=__doc__,
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hostid', dest='host', type=str, action='store',
                        help='CASPER FPGA board hostname or IP.')
    parser.add_argument('-f', '--fpgfile', dest='fpg', type=str, default=fpgfile,
                        help='Specify the fpg file to load')
    parser.add_argument('--noprogram', dest='noprogram', action='store_true',
                        help='Don\'t program FPGA')
    parser.add_argument('-i', '--ipython', dest='ipython', action='store_true',
                        default=False, help='Start IPython at script end.')

    args = parser.parse_args()

    if args.host == None:
        print('Please specify a CASPER FPGA board hostname or IP. \nExiting.')
        sys.exit()
    if os.path.exists(args.fpg) != True:
        print('Could not find %s. Please enter a valid file path. \nExiting.'%args.fpg)
        sys.exit()

    try:

    	fpga = casperfpga.CasperFpga(args.host, log_level='INFO')
    	fpga.is_little_endian = False
    	time.sleep(1)


    	if fpga.is_connected():
            fpga.logger.info('FPGA connected')
    	else:
            fpga.logger.error('ERROR connecting to server %s.\n'%(args.host))
            exit_fail()
    
    
    
    	if not args.noprogram:
            fpga.logger.info('Programming FPGA at %s...' % args.host)
            fpga.upload_to_ram_and_program(args.fpg)
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

    	if args.ipython:
            fpga.logger.info("Opening IPython session")
            import IPython
            IPython.embed()


    except KeyboardInterrupt:
    	exit_clean()
    except Exception as inst:
    	print type(inst)
    	print inst.args
    	print instS
    	exit_fail()
    finally:
	exit_clean()
