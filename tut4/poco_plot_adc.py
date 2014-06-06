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

def read_snap(snapname):
    raw_data = numpy.fromstring(fpga.snapshot_get(snapname, man_trig=True, man_valid=True)['data'], dtype='>i1')
    return raw_data

def drawDataCallback():
    matplotlib.pyplot.clf()
    raw_adc = [[], [], [], []]
    for i in range(0, 4):
        raw_adc[i] = read_snap('snap_adc'+str(i))

        matplotlib.pyplot.subplot(4, 1, i+1)
        matplotlib.pyplot.plot(raw_adc[0], '-o')
        matplotlib.pyplot.grid()
        matplotlib.pyplot.title('ADC ' +str(i))

    fig.canvas.manager.window.after(100, drawDataCallback)
    matplotlib.pyplot.show() 

#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('poco_plot_adc.py <ROACH_HOSTNAME_or_IP> [options]')
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

    ''' 
    for i in range(0, 4):
        raw_data = read_snap('snap_adc'+str(i))
        matplotlib.pyplot.subplot(4,1,i+1)
        matplotlib.pyplot.plot(raw_data, '-o')
    matplotlib.pyplot.title('Raw ADC data')
    matplotlib.pyplot.show()
    '''
    # set up the figure with a subplot for each polarisation to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(4,1,1)

    # start the process
    try:
        fig.canvas.manager.window.after(100, drawDataCallback)
        matplotlib.pyplot.show()
    except:
        pass
    print 'Plotting complete. Exiting...'

except AttributeError:
    pass
except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()

