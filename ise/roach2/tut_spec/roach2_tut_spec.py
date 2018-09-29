#!/usr/bin/env python
'''
This script demonstrates programming an FPGA, configuring a wideband spectrometer and plotting the received data using the Python KATCP library along with the katcp_wrapper distributed in the corr package. Designed for use with TUT3 at the 2009 CASPER workshop.\n
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/
\nAuthor: Jason Manley, November 2009.
Updated by: Tyrone van Balla, October 2015
Updated for CASPER 2016 workshop. Updated to use casperfpga library and for ROACH-2
'''

#TODO: add support for ADC histogram plotting.
#TODO: add support for determining ADC input level 

import casperfpga
import time
import numpy
import struct
import sys
import logging
import pylab
import matplotlib

bitstream = 'tut3_updated.fpg'

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
    #get the data...    
    acc_n = fpga.read_uint('acc_cnt')
    a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
    a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

    interleave_a=[]

    for i in range(1024):
        interleave_a.append(a_0[i])
        interleave_a.append(a_1[i])
    return acc_n, interleave_a 

def plot_spectrum():
    matplotlib.pyplot.clf()
    acc_n, interleave_a = get_data()

    matplotlib.pylab.plot(interleave_a)
    #matplotlib.pylab.semilogy(interleave_a)
    matplotlib.pylab.title('Integration number %i.'%acc_n)
    matplotlib.pylab.ylabel('Power (arbitrary units)')
    matplotlib.pylab.ylim(0)
    matplotlib.pylab.grid()
    matplotlib.pylab.xlabel('Channel')
    matplotlib.pylab.xlim(0,2048)
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, plot_spectrum)

# debug log handler
class DebugLogHandler(logging.Handler):
    """A logger for KATCP tests."""

    def __init__(self,max_len=100):
        """Create a TestLogHandler.
            @param max_len Integer: The maximum number of log entries
                                    to store. After this, will wrap.
        """
        logging.Handler.__init__(self)
        self._max_len = max_len
        self._records = []

    def emit(self, record):
        """Handle the arrival of a log message."""
        if len(self._records) >= self._max_len: self._records.pop(0)
        self._records.append(record)

    def clear(self):
        """Clear the list of remembered logs."""
        self._records = []

    def setMaxLen(self,max_len):
        self._max_len=max_len

    def printMessages(self):
        for i in self._records:
            if i.exc_info:
                print '%s: %s Exception: '%(i.name,i.msg),i.exc_info[0:-1]
            else:    
                if i.levelno < logging.WARNING: 
                    print '%s: %s'%(i.name,i.msg)
                elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                    print '%s: %s'%(i.name,i.msg)
                elif i.levelno >= logging.ERROR: 
                    print '%s: %s'%(i.name,i.msg)
                else:
                    print '%s: %s'%(i.name,i.msg)


#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut3_update.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)/2048,
        help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048, or just under 2 seconds.')
    p.add_option('-g', '--gain', dest='gain', type='int',default=0xffffffff,
        help='Set the digital gain (6bit quantisation scalar). Default is 0xffffffff (max), good for wideband noise. Set lower for CW tones.')
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-f', '--fpgfile', dest='fpg',type='str', default=bitstream,
        help='Specify the bof file to load')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. Run with the -h flag to see all options.\nExiting.'
        exit()
    else:
        roach = args[0] 
    if opts.fpg != '':
        bitstream = opts.fpg

try:
    loggers = []
    lh=DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(roach)),
    fpga = casperfpga.CasperFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(roach)
        exit_fail()

    print '------------------------'
    print 'Programming FPGA with %s...' %bitstream,
    if not opts.skip:
        sys.stdout.flush()
        fpga.upload_to_ram_and_program(bitstream)
        time.sleep(10)
        print 'ok'
    else:
        print 'Skipped.'

    print 'Configuring accumulation period...',
    fpga.write_int('acc_len',opts.acc_len)
    print 'done'

    print 'Resetting counters...',
    fpga.write_int('cnt_rst',1) 
    fpga.write_int('cnt_rst',0) 
    print 'done'

    print 'Setting digital gain of all channels to %i...'%opts.gain,
    if not opts.skip:
        fpga.write_int('gain',opts.gain) #write the same gain for all inputs, all channels
        print 'done'
    else:   
        print 'Skipped.'

    #set up the figure with a subplot to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    # start the process
    fig.canvas.manager.window.after(100, plot_spectrum)
    matplotlib.pyplot.show()
    print 'Plot started.'

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()
#!/usr/bin/env python
'''
This script demonstrates programming an FPGA, configuring a wideband spectrometer and plotting the received data using the Python KATCP library along with the katcp_wrapper distributed in the corr package. Designed for use with TUT3 at the 2009 CASPER workshop.\n
You need to have KATCP and CORR installed. Get them from http://pypi.python.org/pypi/katcp and http://casper.berkeley.edu/svn/trunk/projects/packetized_correlator/corr-0.4.0/
\nAuthor: Jason Manley, November 2009.
Updated by: Tyrone van Balla, October 2015
Updated for CASPER 2016 workshop. Updated to use casperfpga library and for ROACH-2
'''

#TODO: add support for ADC histogram plotting.
#TODO: add support for determining ADC input level 

import casperfpga
import time
import numpy
import struct
import sys
import logging
import pylab
import matplotlib

bitstream = 'tut3_updated.fpg'

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
    #get the data...    
    acc_n = fpga.read_uint('acc_cnt')
    a_0=struct.unpack('>1024l',fpga.read('even',1024*4,0))
    a_1=struct.unpack('>1024l',fpga.read('odd',1024*4,0))

    interleave_a=[]

    for i in range(1024):
        interleave_a.append(a_0[i])
        interleave_a.append(a_1[i])
    return acc_n, interleave_a 

def plot_spectrum():
    matplotlib.pyplot.clf()
    acc_n, interleave_a = get_data()

    matplotlib.pylab.plot(interleave_a)
    #matplotlib.pylab.semilogy(interleave_a)
    matplotlib.pylab.title('Integration number %i.'%acc_n)
    matplotlib.pylab.ylabel('Power (arbitrary units)')
    matplotlib.pylab.ylim(0)
    matplotlib.pylab.grid()
    matplotlib.pylab.xlabel('Channel')
    matplotlib.pylab.xlim(0,2048)
    fig.canvas.draw()
    fig.canvas.manager.window.after(100, plot_spectrum)

# debug log handler
class DebugLogHandler(logging.Handler):
    """A logger for KATCP tests."""

    def __init__(self,max_len=100):
        """Create a TestLogHandler.
            @param max_len Integer: The maximum number of log entries
                                    to store. After this, will wrap.
        """
        logging.Handler.__init__(self)
        self._max_len = max_len
        self._records = []

    def emit(self, record):
        """Handle the arrival of a log message."""
        if len(self._records) >= self._max_len: self._records.pop(0)
        self._records.append(record)

    def clear(self):
        """Clear the list of remembered logs."""
        self._records = []

    def setMaxLen(self,max_len):
        self._max_len=max_len

    def printMessages(self):
        for i in self._records:
            if i.exc_info:
                print '%s: %s Exception: '%(i.name,i.msg),i.exc_info[0:-1]
            else:    
                if i.levelno < logging.WARNING: 
                    print '%s: %s'%(i.name,i.msg)
                elif (i.levelno >= logging.WARNING) and (i.levelno < logging.ERROR):
                    print '%s: %s'%(i.name,i.msg)
                elif i.levelno >= logging.ERROR: 
                    print '%s: %s'%(i.name,i.msg)
                else:
                    print '%s: %s'%(i.name,i.msg)


#START OF MAIN:

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut3_update.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)/2048,
        help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048, or just under 2 seconds.')
    p.add_option('-g', '--gain', dest='gain', type='int',default=0xffffffff,
        help='Set the digital gain (6bit quantisation scalar). Default is 0xffffffff (max), good for wideband noise. Set lower for CW tones.')
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip reprogramming the FPGA and configuring EQ.')
    p.add_option('-f', '--fpgfile', dest='fpg',type='str', default=bitstream,
        help='Specify the bof file to load')
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify a ROACH board. Run with the -h flag to see all options.\nExiting.'
        exit()
    else:
        roach = args[0] 
    if opts.fpg != '':
        bitstream = opts.fpg

try:
    loggers = []
    lh=DebugLogHandler()
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    print('Connecting to server %s... '%(roach)),
    fpga = casperfpga.katcp_fpga.KatcpFpga(roach)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s.\n'%(roach)
        exit_fail()

    print '------------------------'
    print 'Programming FPGA with %s...' %bitstream,
    if not opts.skip:
        sys.stdout.flush()
        fpga.upload_to_ram_and_program(bitstream)
        time.sleep(10)
        print 'ok'
    else:
        print 'Skipped.'

    print 'Configuring accumulation period...',
    fpga.write_int('acc_len',opts.acc_len)
    print 'done'

    print 'Resetting counters...',
    fpga.write_int('cnt_rst',1) 
    fpga.write_int('cnt_rst',0) 
    print 'done'

    print 'Setting digital gain of all channels to %i...'%opts.gain,
    if not opts.skip:
        fpga.write_int('gain',opts.gain) #write the same gain for all inputs, all channels
        print 'done'
    else:   
        print 'Skipped.'

    #set up the figure with a subplot to be plotted
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    # start the process
    fig.canvas.manager.window.after(100, plot_spectrum)
    matplotlib.pyplot.show()
    print 'Plot started.'

except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()
exit_clean()
