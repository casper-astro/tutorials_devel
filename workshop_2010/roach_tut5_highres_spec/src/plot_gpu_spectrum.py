#!/usr/bin/env python2.6

import sys, pylab, math

numchannels=2048


if __name__ == '__main__':
    from optparse import OptionParser

    # set command line options and default values
    p = OptionParser()
    p.set_usage('plot_gpu_spectrum.py [options]')
    p.set_description(__doc__)
    p.set_defaults(channel_select=256)
    p.add_option('-c','--capture_channel', help='record data from a single channel', action='store', type='int', dest='channel_select')
 
    opts, args = p.parse_args(sys.argv[1:])

    # read in the data from the spectrum file
    data=pylab.fromfile('data/channel'+`opts.channel_select`+'_spectrum', dtype='float', sep=' ').reshape(numchannels,2)
    
    # plot the power in log scale
    pylab.yscale('log')
    
    # get an initial power spectrum and plot the power lines as rectangles
    spec_power=[]
    for i in range(0,numchannels):
        spec_power.append(math.pow(data[i][0],2) + math.pow(data[i][1],2))
    pylab.bar(range(0,numchannels),spec_power)
    pylab.draw()
    pylab.ylim(1,max(spec_power))
    pylab.xlim(0,numchannels)
    pylab.show()


