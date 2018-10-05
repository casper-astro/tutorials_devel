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
freq_range_mhz = numpy.linspace(0., 400., 2048)

def set_coeff(coeff,nch):
    quants_coeffs = [coeff] * nch * 2
    quants_coeffs = struct.pack('>{0}H'.format(nch*2),*quants_coeffs)
    for i in range(3):
        fpga.blindwrite('quant{0}_coeffs'.format(i), quants_coeffs)

def get_data():
    #get the data...
    acc_n = fpga.read_uint('acc_num')
    a_0=struct.unpack('>512l',fpga.read('dir_x0_aa_real',512*4,0))
    a_1=struct.unpack('>512l',fpga.read('dir_x1_aa_real',512*4,0))
    b_0=struct.unpack('>512l',fpga.read('dir_x0_bb_real',512*4,0))
    b_1=struct.unpack('>512l',fpga.read('dir_x1_bb_real',512*4,0))
    c_0=struct.unpack('>512l',fpga.read('dir_x0_cc_real',512*4,0))
    c_1=struct.unpack('>512l',fpga.read('dir_x1_cc_real',512*4,0))

    interleave_a=[]
    interleave_b=[]
    interleave_c=[]

    for i in range(512):
        interleave_a.append(a_0[i])
        interleave_a.append(a_1[i])
        interleave_b.append(b_0[i])
        interleave_b.append(b_1[i])
        interleave_c.append(c_0[i])
        interleave_c.append(c_1[i])
    return acc_n, numpy.array(interleave_a,dtype=numpy.float), numpy.array(interleave_b,dtype=numpy.float), numpy.array(interleave_c,dtype=numpy.float)

def plot_spectrum():
    matplotlib.pyplot.clf()
    acc_n, interleave_a, interleave_b, interleave_c = get_data()
    #interleave_a = 10*numpy.log10(interleave_a/(numpy.max(interleave_a)))
    interleave_a = 10*numpy.log10(interleave_a)
    interleave_b = 10*numpy.log10(interleave_b)
    interleave_c = 10*numpy.log10(interleave_c)
    matplotlib.pylab.plot(freq_range_mhz, interleave_a)
    matplotlib.pylab.plot(freq_range_mhz, interleave_b,'r')
    matplotlib.pylab.plot(freq_range_mhz, interleave_c,'g')
    matplotlib.pylab.title('Integration number %i.'%acc_n)
    matplotlib.pylab.ylabel('Power (dB)')
    #matplotlib.pylab.ylim(0)
    matplotlib.pylab.grid()
    matplotlib.pylab.xlabel('Freq (MHz)')
    matplotlib.pylab.xlim(freq_range_mhz[0], freq_range_mhz[-1])
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, plot_spectrum)


#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser


    p = OptionParser()
    p.set_usage('spectrometer.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_argument('-n', '--nchannel', dest='nch',type=int, default=1024,
		help='The number of frequency channel. Default is 1024.')
    p.add_argument('-c', '--coeff', dest='coeff', type=int, default=1000,
		help='Set the coefficients in quantisation (4bit quantisation scalar).')
    p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)/2048,
        help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048, or just under 2 seconds.')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a SNAP board. Run with the -h flag to see all options.\nExiting.'
        exit()
    else:
        snap = args[0] 

try:

    print('Connecting to server %s on port %i... '%(snap,katcp_port)),
    fpga = casperfpga.CasperFpga(snap)
    time.sleep(0.2)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(snap,katcp_port)
        exit_fail()


    print 'Configuring accumulation period...',
    sys.stdout.flush()
    if args.acc_len:
        fpga.write_int('acc_len',opts.acc_len)
    print 'done'

    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<16) #clip reset
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<17) #arm
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<18) #software trigger
    fpga.write_int('ctrl',0) 
    fpga.write_int('ctrl',1<<18) #issue a second trigger
    fpga.write_int('ctrl',0) 
    time.sleep(0.1)

    if args.coeff:
        print 'Configuring quantisation coefficients with parameter {0}...'.format(args.coeff),
        sys.stdout.flush()
        set_coeff(args.coeff,args.nch)
        print 'done'

   #set up the figure with a subplot to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    # start the process
    fig.canvas.manager.window.after(100, plot_spectrum)
    matplotlib.pyplot.show()
    print 'Plot started.'


except KeyboardInterrupt:
    exit()

exit()

