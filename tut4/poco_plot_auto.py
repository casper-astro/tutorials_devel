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

def drawData_animate():
    #matplotlib.pyplot.ion()
    fig= matplotlib.pyplot.figure(num=1)
    acc_n,interleave_a,interleave_b,interleave_c,interleave_d = get_data()

    #matplotlib.pyplot.subplot(411)
    ax1 = fig.add_subplot(4,1,1)
    ax1.set_title('Integration number %i \nAA'%acc_n)	
    if ifch:
        plot1, = matplotlib.pyplot.semilogy(interleave_a)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot1, = matplotlib.pyplot.semilogy(xaxis,interleave_a)
    matplotlib.pyplot.grid()
    #matplotlib.pyplot.title('Integration number %i \nAA'%acc_n)
    matplotlib.pyplot.ylabel('Power (arbitrary units)')

    ax2 = fig.add_subplot(412)
    if ifch:
        plot2, = matplotlib.pyplot.semilogy(interleave_b)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot2, = matplotlib.pyplot.semilogy(xaxis,interleave_b)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('BB')

    ax3 = fig.add_subplot(413)
    if ifch:
        plot3, = matplotlib.pyplot.semilogy(interleave_c)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot3, = matplotlib.pyplot.semilogy(xaxis,interleave_c)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('CC')

    ax4 = fig.add_subplot(414)
    if ifch:
        plot4, = matplotlib.pyplot.semilogy(interleave_d)
        matplotlib.pyplot.xlim(0,1024)
        matplotlib.pyplot.xlabel('Channel')
    else:
        plot4, = matplotlib.pyplot.semilogy(xaxis,interleave_d)
        matplotlib.pyplot.xlabel('Frequency')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('DD')

    def init():
	if ifch:
		plot1.set_ydata(numpy.zeros(1,1024))
		plot2.set_ydata(numpy.zeros(1,1024))
		plot3.set_ydata(numpy.zeros(1,1024))
		plot4.set_ydata(numpy.zeros(1,1024))
	else:
		plot1.set_ydata(numpy.zeros(1,size(xaxis)+1))
		plot2.set_ydata(numpy.zeros(1,size(xaxis)+1))
		plot3.set_ydata(numpy.zeros(1,size(xaxis)+1))
		plot4.set_ydata(numpy.zeros(1,size(xaxis)+1))
	return ax, plot1, plot2, plot3, plot4,

    def update():
	acc_n,interleave_a,interleave_b,interleave_c,interleave_d = get_data()
	plot1.set_ydata(interleave_a)
	ax.set_title('Integration number %i \nAA'%acc_n)
	plot2.set_ydata(interleave_b)
	plot3.set_ydata(interleave_c)
	plot4.set_ydata(interleave_d)
	return ax, plot1, plot2, plot3, plot4,

    ani = matplotlib.animation.FuncAnimation(fig, update, init_func = init, interval = 10, blit = True)
    matplotlib.pyplot.show()



def drawDataLoop():
    matplotlib.pyplot.ion()
    fig = matplotlib.pyplot.figure(num = 1, figsize=(18,12))
    acc_n,interleave_a,interleave_b,interleave_c,interleave_d = get_data()


    ax1 = fig.add_subplot(411)
    if ifch:
        plot1, = matplotlib.pyplot.semilogy(interleave_a)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot1, = matplotlib.pyplot.semilogy(xaxis,interleave_a)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.title('Auto-Correlation \nIntegration :%ith cycle \nAA (antenna A)'%acc_n)
    matplotlib.pyplot.ylabel('Power (arbitrary units)')

    matplotlib.pyplot.subplot(412)
    if ifch:
        plot2, = matplotlib.pyplot.semilogy(interleave_b)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot2, = matplotlib.pyplot.semilogy(xaxis,interleave_b)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('BB (Antenna B)')

    matplotlib.pyplot.subplot(413)
    if ifch:
        plot3, = matplotlib.pyplot.semilogy(interleave_c)
        matplotlib.pyplot.xlim(0,1024)
    else:
        plot3, = matplotlib.pyplot.semilogy(xaxis,interleave_c)
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('CC (Antenna C)')

    matplotlib.pyplot.subplot(414)
    if ifch:
        plot4, = matplotlib.pyplot.semilogy(interleave_d)
        matplotlib.pyplot.xlim(0,1024)
        matplotlib.pyplot.xlabel('Channel')
    else:
        plot4, = matplotlib.pyplot.semilogy(xaxis,interleave_d)
        matplotlib.pyplot.xlabel('Frequency (MHz)')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.ylabel('Power (arbitrary units)')
    matplotlib.pyplot.title('DD (Antenna D)')

    while(1):
	try:
		acc_n,interleave_a,interleave_b,interleave_c,interleave_d = get_data()
		ax1.set_title('Auto-Correlation \nIntegration :%ith cycle \nAA (antenna A)'%acc_n)
		plot1.set_ydata(interleave_a)
		matplotlib.pyplot.hold(False)
		matplotlib.pyplot.draw()
		matplotlib.pyplot.hold(True)
		plot2.set_ydata(interleave_b)
		matplotlib.pyplot.hold(False)
		matplotlib.pyplot.draw()
		matplotlib.pyplot.hold(True)
		plot3.set_ydata(interleave_c)
		matplotlib.pyplot.hold(False)
		matplotlib.pyplot.draw()
		matplotlib.pyplot.hold(True)
		plot4.set_ydata(interleave_d)
		matplotlib.pyplot.hold(False)
		matplotlib.pyplot.draw()
		matplotlib.pyplot.hold(True)
		time.sleep(1)
	except (KeyboardInterrupt, SystemExit):
		print '\nKeyboard interrupt caught, exiting...\n'
		break


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

    #matplotlib.pyplot.show()
    fig.canvas.manager.window.after(1000, drawDataCallback)


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
    #fig = matplotlib.pyplot.figure()
    #ax = fig.add_subplot(4,1,1)

    # start the process
    #fig.canvas.manager.window.after(1000, drawDataCallback)
    #matplotlib.pyplot.show()
    drawDataLoop()
    #drawData_animate()
    print 'Exiting...'


except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

