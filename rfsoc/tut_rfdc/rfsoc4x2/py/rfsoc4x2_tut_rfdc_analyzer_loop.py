import sys, os, time
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import casperfpga

from rfsoc4x2_tut_rfdc_utils import *

def plot_spectrum(fpgas, cx=False):
  # prepare the figure to plot all 4 adc inputs
  PLT_WIDTH = 2
  PLT_DEPTH = 2
  fig, ax = plt.subplots(PLT_DEPTH, PLT_WIDTH, sharey = 'row', sharex='col')
  lines = []

  # variables for our setup
  fs = 3932.16/2
  Nfft = 2**11
  df = fs/Nfft
  fbins = np.arange(-Nfft//2, Nfft//2)
  faxis = fbins*df

  Nant = 4
  samp_per_word = 8
  Nsamps = 16384*samp_per_word # 16384 word deep memory and 8 samples per word

  if cx:
    # real/imag parts are interlaved on the interfaces for 28/48dr rfsoc. These
    # strings match the name of the snapshot block in the simulink design
    el_list = ['m00', 'm01', 'm02', 'm03', 'm20', 'm21', 'm22', 'm23']
    # add NCO offset
    faxis = faxis + fs/2
  else:
    # strings match the name of the snapshot block in the simulink design
    el_list = ['m00', 'm02', 'm20', 'm22']

  # get data to plot once before going into animation loop (lazily axes init and lines)
  raw = get_rfsoc_data(fpga)
  # cx is set to false here because we are not having this function build cx
  # samples. If the snapshot block captured interleaved complex data we would
  # could use 'True' here. But because the snapshots in this example are setup
  # to capture one interface stream we need to form complex words elsewhere.
  X = form_data_mat(raw, el_list, Nsamps, samp_per_word, cx=False)

  if cx: # combine real/imag from the two interfaces
    ytmp = np.reshape(X, (int(Nant*2), Nsamps//Nfft, Nfft))
    y = np.zeros((Nant, Nsamps//Nfft, Nfft), dtype=np.complex128)
    for i in range(0, Nant):
      y[i,:,:] = ytmp[2*i,:,:] + 1j*ytmp[2*i+1,:,:]
  else:
    y = np.reshape(X, (Nant, Nsamps//Nfft, Nfft))

  Y = fft.fft(y, Nfft, axis=2)
  Ypsd = np.mean(np.real(Y*np.conj(Y)), axis=1)

  # make first plot and setup axes, labels, grid and save the lines to use later
  for i in range(0, PLT_DEPTH):
    for j in range(0, PLT_WIDTH):
      idx = (i*PLT_WIDTH) + j
      cur_ax = ax[i, j]
      line, = cur_ax.plot(faxis, 10*np.log10(fft.fftshift(Ypsd[idx,:])), '-')
      lines.append(line)
      cur_ax.grid();
      cur_ax.set_title('port {:d}'.format(idx))

      if i==(PLT_DEPTH-1):
        cur_ax.set_xlabel('freq')
      if j==0:
        cur_ax.set_ylabel('PSD (dB arb.)')
  fig.suptitle('PSD')

  # plot update function
  def update(frame, *fargs):
    raw = get_rfsoc_data(fpgas)
    # cx is set to false here because we are not having this function build cx
    # samples. If the snapshot block captured interleaved complex data we would
    # could use 'True' here. But because the snapshots in this example are setup
    # to capture one interface stream we need to form complex words elsewhere.
    X = form_data_mat(raw, el_list, Nsamps, samp_per_word, cx=False)

    if cx: # combine real/imag from the two interfaces
      ytmp = np.reshape(X, (int(Nant*2), Nsamps//Nfft, Nfft))
      y = np.zeros((Nant, Nsamps//Nfft, Nfft), dtype=np.complex128)
      for i in range(0, Nant):
        y[i,:,:] = ytmp[2*i,:,:] + 1j*ytmp[2*i+1,:,:]
    else:
      y = np.reshape(X, (Nant, Nsamps//Nfft, Nfft))

    Y = fft.fft(y, Nfft, axis=2)
    Ypsd = np.mean(np.real(Y*np.conj(Y)), axis=1)

    # update line data
    for i, line in enumerate(lines):
      line.set_ydata(10*np.log10(fft.fftshift(Ypsd[i,:])))

  v = anim.FuncAnimation(fig, update, frames=1, repeat=True, fargs=None, interval=100)
  plt.show()


if __name__=="__main__":
  from optparse import OptionParser

  p = OptionParser()
  p.set_usage('rfsoc4x2_tut_rfdc_analyzer_loop.py <HOSTNAME_or_IP> cx|real [options]')
  p.set_description(__doc__)
  p.add_option('-s', '--skip', dest='skip', action='store_true',
      help='Skip programming and begin to plot data')
  p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='',
      help='Specify the fpg file to load')

  opts, args = p.parse_args(sys.argv[1:])
  if len(args) < 2:
    print('Specify a hostname or IP for your casper platform. And either cx|real to indicate the type of spectrometer design.\n'
          'Run with the -h flag to see all options.')
    exit()
  else:
    hostname = args[0]
    mode_str = args[1]
    print("mode=", mode_str)
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
      fpg_prebuilt = '../../prebuilt/rfsoc4x2/rfsoc4x2_tut_rfdc_cx.fpg'
    else:
      fpg_prebuilt = '../../prebuilt/rfsoc4x2/rfsoc4x2_tut_rfdc_real.fpg'

    print('using prebuilt fpg file at %s' % fpg_prebuilt)
    bitstream = fpg_prebuilt

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

  try:
    plot_spectrum(fpga, cx=mode)
  except KeyboardInterrupt:
    exit()
