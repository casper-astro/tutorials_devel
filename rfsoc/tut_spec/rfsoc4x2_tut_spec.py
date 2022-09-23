import sys, time, struct
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import casperfpga

def get_vacc_data(fpga, nchannels=8, nfft=2048):
  acc_n = fpga.read_uint('acc_cnt')
  chunk = nfft//nchannels

  raw = np.zeros((nchannels, chunk))
  for i in range(nchannels):
    raw[i,:] = struct.unpack('>{:d}Q'.format(chunk), fpga.read('q{:d}'.format((i+1)),chunk*8,0))

  interleave_q = []
  for i in range(chunk):
    for j in range(nchannels):
      interleave_q.append(raw[j,i])

  return acc_n, np.array(interleave_q, dtype=np.float64)

def plot_spectrum(fpga, cx=True, num_acc_updates=None):

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.grid();

  fs = 3932.16/2
  if cx:
    print('complex design')
    Nfft = 2**11
    fbins = np.arange(-Nfft//2, Nfft//2)
    nchannels = 8
    df = fs/Nfft
    faxis = fbins*df + fs/2
  else:
    print('real design')
    Nfft = 2**12
    fbins = np.arange(0, Nfft//2)
    df = fs/Nfft
    nchannels = 4
    faxis = fbins*df

  if cx:
    acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft)
    line, = ax.plot(faxis,10*np.log10(fft.fftshift(spectrum)),'-')
  else:
    acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft//2)
    line, = ax.plot(faxis,10*np.log10(spectrum),'-')

  def update(frame, *fargs):
    if cx:
      acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft)
      line.set_ydata(10*np.log10(fft.fftshift(spectrum)))
    else:
      acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft//2)
      line.set_ydata(10*np.log10(spectrum))

    ax.set_title('acc num: %d' % acc_n)

  v = anim.FuncAnimation(fig, update, frames=1, repeat=True, fargs=None, interval=1000)
  plt.show()

if __name__=="__main__":
  from optparse import OptionParser

  p = OptionParser()
  p.set_usage('rfsoc4x2_tut_spec.py <HOSTNAME_or_IP> cx|real [options]')
  p.set_description(__doc__)
  p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)//2048,
      help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048')
  p.add_option('-s', '--skip', dest='skip', action='store_true',
      help='Skip programming and begin to plot data')
  p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='',
      help='Specify the fpg file to load')
  p.add_option('-a', '--adc', dest='adc_chan_sel', type=int, default=0,
      help='adc input to select values are 0,1,2, or 3')

  opts, args = p.parse_args(sys.argv[1:])
  if len(args) < 2:
    print('Specify a hostname or IP for your casper platform. And either cx|real to indicate the type of spectrometer design.\n'
          'Run with the -h flag to see all options.')
    exit()
  else:
    hostname = args[0]
    mode_str = args[1]
    if mode_str=='cx':
      mode = 1
    elif mode_str=='real':
      mode = 0
    else:
      print('operation mode not recognized, must be "cx" or "real"')
      exit()

  if opts.fpgfile != '':
    bitstream = opts.fpgfile
  else:
    if mode == 1:
      fpg_prebuilt = './prebuilt/rfsoc4x2_tut_spec_cx.fpg'
    else:
      fpg_prebuilt = './prebuilt/rfsoc4x2_tut_spec.fpg'

    print('using prebuilt fpg file at %s' % fpg_prebuilt)
    bitstream = fpg_prebuilt

  if opts.adc_chan_sel < 0 or opts.adc_chan_sel > 3:
    print('adc select must be 0,1,2, or 3')
    exit()

  print('Connecting to %s... ' % (hostname))
  fpga = casperfpga.CasperFpga(hostname)
  time.sleep(0.2)

  if not opts.skip:
    print('Programming FPGA with %s...'% bitstream)
    fpga.upload_to_ram_and_program(bitstream)
    print('done')
  else:
    fpga.get_system_information()
    print('skip programming fpga...')

  print('Configuring accumulation period...')
  fpga.write_int('acc_len',opts.acc_len)
  time.sleep(0.1)
  print('done')

  print('setting capture on adc port %d' % opts.adc_chan_sel)
  fpga.write_int('adc_chan_sel', opts.adc_chan_sel)
  time.sleep(0.1)
  print('done')

  print('Resetting counters...')
  fpga.write_int('cnt_rst',1) 
  fpga.write_int('cnt_rst',0) 
  time.sleep(2)
  print('done')

  try:
    plot_spectrum(fpga, cx=mode)
  except KeyboardInterrupt:
    exit()
