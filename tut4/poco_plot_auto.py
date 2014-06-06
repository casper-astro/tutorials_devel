#!/usr/bin/env python
'''
This script demonstrates grabbing data off an already configured FPGA and plotting it using Python. Designed for use with CASPER workshop Tutorial 4.
\n\n 
Author: Jason Manley, August 2009.
Modified: Aug 2012, Nie Jun
'''

#TODO: add support for coarse delay change
#TODO: add support for ADC histogram plotting.
#TODO: add support for determining ADC input level 

import corr,time,numpy,struct,sys,logging,pylab,matplotlib

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

def get_data():
    acc_n = fpga.read_uint('acc_num')
    print 'Grabbing integration number %i'%acc_n

    a_0=struct.unpack('>512l',fpga.read('dir_x0_aa_real',2048,0))
    a_1=struct.unpack('>512l',fpga.read('dir_x1_aa_real',2048,0))
    b_0=struct.unpack('>512l',fpga.read('dir_x0_bb_real',2048,0))
    b_1=struct.unpack('>512l',fpga.read('dir_x1_bb_real',2048,0))
    c_0=struct.unpack('>512l',fpga.read('dir_x0_cc_real',2048,0))
    c_1=struct.unpack('>512l',fpga.read('dir_x1_cc_real',2048,0))
    d_0=struct.unpack('>512l',fpga.read('dir_x0_dd_real',2048,0))
    d_1=struct.unpack('>512l',fpga.read('dir_x1_dd_real',2048,0))

    interleave_a=[]
    interleave_b=[]
    interleave_c=[]
    interleave_d=[]

    for i in range(512):
        interleave_a.append(a_0[i])
        interleave_a.append(a_1[i])
        interleave_b.append(b_0[i])
        interleave_b.append(b_1[i])
        interleave_c.append(c_0[i])
        interleave_c.append(c_1[i])
        interleave_d.append(d_0[i])
        interleave_d.append(d_1[i])

    return acc_n,interleave_a,interleave_b,interleave_c,interleave_d

def drawDataCallback():
    matplotlib.pyplot.clf()
    acc_n,interleave_a,interleave_b,interleave_c,interleave_d = get_data()

    matplotlib.pyplot.subplot(411)
    if ifch:
        matplotlib.pyplot.semilogy(interleave_a)
        matplotlib.pyplot.xlim(0,1024)
    else:
        matplotlib.pyplot.semilogy(xaxis,interleave_a)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title('Integration number %i \nAA'%acc_n)
    matplotlib.pyplot.ylabel('Power (arbitrary units)')

    matplotlib.pyplot.subplot(412)
    if ifch:
        matplotlib.pyplot.semilogy(interleave_b)
        matplotlib.pyplot.xlim(0,1024)
    else:
        matplotlib.pyplot.semilogy(xaxis,interleave_b)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('BB')

    matplotlib.pyplot.subplot(413)
    if ifch:
        matplotlib.pyplot.semilogy(interleave_c)
        matplotlib.pyplot.xlim(0,1024)
    else:
        matplotlib.pyplot.semilogy(xaxis,interleave_c)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('CC')

    matplotlib.pyplot.subplot(414)
    if ifch:
        matplotlib.pyplot.semilogy(interleave_d)
        matplotlib.pyplot.xlim(0,1024)
        matplotlib.pyplot.xlabel('Channel')
    else:
        matplotlib.pyplot.semilogy(xaxis,interleave_d)
        matplotlib.pyplot.xlabel('Frequency')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('DD')

    fig.canvas.manager.window.after(100, drawDataCallback)
    matplotlib.pyplot.show()

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut3.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-C','--channel',dest='ch',action='store_true',
        help='Set plot with channel number or frequency.')
    p.add_option('-f','--frequency',dest='fr',type='float',default=400.0,
        help='Set plot max frequency.(If -c sets to False)')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. \nExiting.'
        exit()
    else:
        roach = args[0]

    if opts.ch !=None:
        ifch = opts.ch
    else:
        ifch = False

    if ifch == False:
        if opts.fr != '':
            maxfr = opts.fr
        else:
            maxfr = 400.0
        xaxis = numpy.arange(0.0, maxfr, maxfr*1./1024)

# What to be shown on X axis while ploting
# ifch means if the X axis is channel number

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


    prev_integration = fpga.read_uint('acc_num')


    # set up the figure with a subplot for each polarisation to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(4,1,1)

    # start the process
    fig.canvas.manager.window.after(100, drawDataCallback)
    matplotlib.pyplot.show()
    print 'Exiting...'

except AttributeError:
    pass
except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

