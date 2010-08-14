#!/usr/bin/env python2.6

import struct, socket, sys ,corr, logging, time, re
import math

# default boffile
defaultbof='gpu_spec.bof'

# define katcp port to connect
katcp_port=7147

# fft configuration
# shifting schedule
fft_shift=0xfffffffe

# simulink model file configuration parameters
numchannels=512
bram_size=2048

# interleave the odd and even channels
def interleave(*args):
    retlist=[]
    for idx in range(0, max(len(arg) for arg in args)):
        for arg in args:
            try:
                retlist.append(arg[idx])
            except IndexError:
                continue
    return retlist

# get the data from the debugging brams recording the entire spectrum
def get_fft_brams(bramlen):
    # read in the even and odd channels
    fftscope_even_re_pol0 = struct.unpack('>'+`bramlen`+'l',fpga.read('scope_output1_bram',bramlen*4))
    fftscope_even_im_pol0 = struct.unpack('>'+`bramlen`+'l',fpga.read('scope_output2_bram',bramlen*4))
    fftscope_odd_re_pol0 = struct.unpack('>'+`bramlen`+'l',fpga.read('scope_output3_bram',bramlen*4))
    fftscope_odd_im_pol0 = struct.unpack('>'+`bramlen`+'l',fpga.read('scope_output4_bram',bramlen*4))
    # interleave the even and odd channels to get a continuous spectrum
    fftscope_re_pol0 = interleave(fftscope_even_re_pol0,fftscope_odd_re_pol0)
    fftscope_im_pol0 = interleave(fftscope_even_im_pol0,fftscope_odd_im_pol0)

    return fftscope_re_pol0, fftscope_im_pol0

# calculate the power spectrum from the debugging brams
def get_fft_brams_power():
    retlist_pol0=[]
    fftscope_re_pol0, fftscope_im_pol0 = get_fft_brams(numchannels/2)
    for i in range(0,numchannels):
        retlist_pol0.append(math.pow(fftscope_re_pol0[i],2) + math.pow(fftscope_im_pol0[i],2))
    return retlist_pol0

# plot the data from the debugging registers that record the entire spectrum
def plot_fft_brams():
    import pylab
    run = True
    
    # turn on live updating
    pylab.ion()
    # plot the power in log scale
    pylab.yscale('log')
    
    # get an initial power spectrum and plot the power lines as rectangles
    fftscope_power = get_fft_brams_power()
    fftscope_power_line=pylab.bar(range(0,numchannels),fftscope_power)
    
    pylab.ylim(1,1000000)
    
    # plot until an interrupt is received
    # for i in range(1,10):
    while(run):
        try:
            # read in a new spectrum
            fftscope_power = get_fft_brams_power()
            
            # update the rectangles based on the new power spectrum
            for j in range(0,numchannels):
                fftscope_power_line[j].set_height(fftscope_power[j])
            #update the plot
            pylab.draw()
        except KeyboardInterrupt:
            run = False
    
    # after stopping the liveupdate leave the plot up until the user is done with it
    raw_input('Press enter to quit: ')
            
    pylab.cla() 


def exit_fail():
    print 'FAILURE DETECTED. Log entries:\n',lh.printMessages()
    try:
        fpga.stop()
    except: pass
    raise
    #exit()

def exit_clean():
    #print 'NO FAILURES'
    try:
        for f in fpgas: f.stop()
    except: pass
    #exit()
    
# capture and record data from a single channel
def capture_channel(channel):
    fpga.write_int('channel_select',channel)
    # toggle the capture register
    fpga.write_int('start_capture',1)
    time.sleep(1)
    fpga.write_int('start_capture',0)
    
    FILE=open('data/channel'+`channel`+'_out','w')
    
    # read in the real and imaginary data
    realdata = struct.unpack('>'+`bram_size`+'l',fpga.read('re_channel_bram',bram_size*4))
    imdata = struct.unpack('>'+`bram_size`+'l',fpga.read('im_channel_bram',bram_size*4))
    
    # write out the real and imaginary data to a file
    for i in range(0,bram_size):
        FILE.write('%d %d\n'%(realdata[i], imdata[i]))

if __name__ == '__main__':
    from optparse import OptionParser

    # set command line options and default values
    p = OptionParser()
    p.set_usage('gpu_spec_init.py <ROACH_HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.set_defaults(boffile=defaultbof)
    p.set_defaults(plot_mode=False)
    p.set_defaults(channel_select=256)
    p.add_option('-f','--file', help='use specified boffile', action='store', type='string', dest='boffile')
    p.add_option('-p','--plot', help='plot the spectrum', action='store_true', dest='plot_mode')
    p.add_option('-c','--capture_channel', help='record data from a single channel', action='store', type='int', dest='channel_select')
 
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'No ROACH board specified. Defaulting to ROACH01'
        roach = 'roach01'
    else:
        roach = args[0]

try:
    lh=corr.log_handlers.DebugLogHandler()
    global logger 
    logger = logging.getLogger(roach)
    logger.addHandler(lh)
    logger.setLevel(10)

    # connect to the fpga
    print('Connecting to server %s on port %i... '%(roach,katcp_port))
    fpga = corr.katcp_wrapper.FpgaClient(roach, katcp_port, timeout=10,logger=logger)
    time.sleep(1)

    if fpga.is_connected():
        print 'ok\n'
    else:
        print 'ERROR connecting to server %s on port %i.\n'%(roach,katcp_port)
        exit_fail()
        
    # program the fpga with selected boffile
    logger.debug('Programming the fpga with %s'%(opts.boffile))   
    print 'Programming the fpga with %s'%(opts.boffile)  
    fpga.progdev(opts.boffile)
    time.sleep(10)
    print 'Programming complete'
    
    # initialize the configuration registers
    fpga.write_int('fft_shift',fft_shift)
    
    # capture a single channel to a file
    capture_channel(opts.channel_select)

    # if plotting was selected on the command line open up a live updating plot
    if opts.plot_mode == True:
        plot_fft_brams();



except KeyboardInterrupt:
    exit_clean()
except:
    exit_fail()

exit_clean()
