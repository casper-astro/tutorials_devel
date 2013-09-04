#!/usr/bin/env python
'''
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/

\nAuthor: Jack Hickish, 2013
'''

import corr,time,numpy,struct,sys,logging,pylab,matplotlib,select

bitstream = 'tut6.bof'
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

def plot_tl():
    matplotlib.pyplot.clf()
    red,amber,green = get_leds()
    if heardEnter():
        print "Heard Enter"
        fpga.write_int('toggle',0)
        fpga.write_int('toggle',1)
        fpga.write_int('toggle',0)
    rect = matplotlib.pyplot.Rectangle((0.375,0.125),0.25,0.75,ec='k',fc='k')
    matplotlib.pyplot.gca().add_patch(rect)
    if red: drawCircle((0.5,0.75),c='r')
    else: drawCircle((0.5,0.75),c='w')
    if amber: drawCircle((0.5,0.5),c='y')
    else: drawCircle((0.5,0.5),c='w')
    if green: drawCircle((0.5,0.25),c='g')
    else: drawCircle((0.5,0.25),c='w')

    fig.canvas.draw()
    fig.canvas.manager.window.after(100, plot_tl)

def drawCircle(loc=(0,0),radius=0.125,c='k'):
    ax = matplotlib.pyplot.gca()
    circ = matplotlib.pyplot.Circle(loc,radius=0.125,color=c,ec='k')
    ax.add_patch(circ)

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

def get_leds():
    leds = fpga.read_int('leds')
    red   = bool(leds&(1<<0))
    amber = bool(leds&(1<<1))
    green = bool(leds&(1<<2))
    return red,amber,green

#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser


    p = OptionParser()
    p.set_usage('tut6.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA. Default = False')
    p.add_option('-b', '--bof', dest='boffile',type='str', default='',
        help='Specify the bof file to load')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. Run with the -h flag to see all options.\nExiting.'
        exit()
    else:
        roach = args[0] 
    if opts.boffile != '':
        bitstream = opts.boffile

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
    print 'Programming FPGA with %s...' %bitstream,
    if not opts.skip:
        fpga.progdev(bitstream)
        print 'done'
    else:
        print 'Skipped.'

    print 'Toggling reset...',
    fpga.write_int('rst',1)
    fpga.write_int('rst',0)
    print 'done'

    #set up the figure with a subplot to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    # start the process
    print "Hit Enter to toggle lights"
    print "(make sure you have keyboard focus on the terminal, not the plot)"
    fig.canvas.manager.window.after(100, plot_tl)
    matplotlib.pyplot.show()


except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

