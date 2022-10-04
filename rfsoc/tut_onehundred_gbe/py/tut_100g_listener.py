import sys, time
import struct
import numpy as np
import matplotlib.pyplot as plt
import casperfpga

from multiprocessing.connection import Listener
from numpy import fft

# hardcoded parameters from deteremined by design
Nbyte_per_time = 4  # bytes per sample (I/Q time sample)
Nbyte_hdr = 6
Nbyte_per_word = 64
Nword_per_pkt = 128

PAYLOAD_SIZE = Nword_per_pkt*Nbyte_per_word
PKT_SIZE = PAYLOAD_SIZE + Nbyte_hdr
Nt = PAYLOAD_SIZE//Nbyte_per_time  # time samples per packet


if __name__ == "__main__":
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('tut_100g_listener.py <HOSTNAME_or_IP> [options]')
    p.set_description(__doc__)
    p.add_option('-n', '--numpkt', dest='numpkt', type='int', default=2**8,
        help='Set the number of packets captured in sequence and then sent to listener. Must be power of 2. default is 2**8')
    p.add_option('-s', '--skip', dest='skip', action='store_true',
        help='Skip programming and begin to plot data')
    p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='',
        help='Specify the fpg file to load')
    p.add_option('-a', '--adc', dest='adc_chan_sel', type=int, default=0,
        help='adc input to select values are 0,1,2, or 3. deafult is 0')

    opts, args = p.parse_args(sys.argv[1:])
    if len(args) < 1:
      print('Specify a hostname or IP for your casper platform. Run with the -h flag to see all options.\n')
      sys.exit()
    else:
      hostname = args[0]

    if opts.fpgfile != '':
      bitstream = opts.fpgfile
    else:
      fpg_prebuilt = '../prebuilt/rfsoc4x2/rfsoc4x2_tut_100g_stream_rfdc.fpg'

      print('using prebuilt fpg file at {:s}'.format(fpg_prebuilt))
      bitstream = fpg_prebuilt

    if opts.numpkt:
        pwr = np.floor(np.log2(opts.numpkt))
        if pwr > 16:
            print("requested to capture more than {:d} packets, this would require too much memory... not going to try.")
            sys.exit()
        NPKT = int(2**pwr)
    else:
        NPKT = int(2**8)

    if opts.adc_chan_sel < 0 or opts.adc_chan_sel > 3:
      print('adc select must be 0,1,2, or 3')
      sys.exit()

    # connect and configure CASPER fpga platform
    print('Connecting to {:s}... '.format(hostname))
    fpga = casperfpga.CasperFpga(hostname)
    time.sleep(0.2)

    if not opts.skip:
      print('Programming FPGA with {:s}...'.format(bitstream))
      fpga.upload_to_ram_and_program(bitstream)
      print('done')
    else:
      fpga.get_system_information()
      print('skip programming fpga...')

    print('setting capture on adc port {:d}'.format(opts.adc_chan_sel))
    fpga.write_int('adc_chan_sel', opts.adc_chan_sel)
    time.sleep(0.1)

    # setup waiting for 100g sender to connect
    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    print('waiting for catcher to connect...')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)


    print("\nsending parameters to catcher...")
    print("Number packets per sequence: {:d}".format(NPKT))
    print("Number time samples per pkt: {:d}".format(Nt))
    conn.send([NPKT, PKT_SIZE, Nt])

    payload_fmt_str = "{:d}h".format(Nt*2)

    # setup from parameters before continuing
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    line1 = None
    plt_cnt = 0

    Nfft = 2**10
    fbins = np.arange(-Nfft//2, Nfft//2)
    fs = 3932.16/2
    df = fs/Nfft
    faxis = df*fbins + fs/2

    while True:
        pkts = []
        conn.send('ready')

        i = 0
        while i < NPKT:
            r = conn.recv()
            if r == 'close':
                print("recv'd 'close'... exiting...")
                listener.close()
                sys.exit()
            pkts.append(r)
            i+=1

        cnts = np.zeros((NPKT,))
        data = np.zeros((NPKT, Nt*2), dtype=int)
        vld = 0
        for i, p in enumerate(pkts):
            # discard runt packets
            if len(p) < PKT_SIZE:
                print("caught runt (unexpected) packet... discarding...")
                continue

            eth_hdr = p[0:14]   # [dstmac, srcmac, ethtype]
            ip_hdr  = p[14:34]  # [..., srcip, dstip]
            udp_hdr = p[34:42]  # [src port, dst port, len]
            pkt_hdr = p[42:106] # 64 byte header [sequence_cnt, zeros...]

            payload = p[106:]   # packet payload data

            src_ip = struct.unpack("4B", ip_hdr[-8:-4])
            dst_ip = struct.unpack("4B", ip_hdr[-4:])

            # simple pkt filtering on destination IP (10.17.16.60)
            src_ip = (src_ip[0] << 24) | (src_ip[1] << 16) | (src_ip[2] << 8) | (src_ip[3])
            src_ip_expected = 10*(2**24) + 17*(2**16) + 16*(2**8) + 60
            if src_ip != src_ip_expected:
                print("caught packet with unexpected ip address... discarding...")

            # get packet sequence counts
            c = struct.unpack("L", pkt_hdr[0:8])
            d = struct.unpack(payload_fmt_str, payload)
            cnts[vld]   = c[0]
            data[vld,:] = np.array(d)
            # can do more filtering based on other parameters (e.g., warn on dropped counts)
            vld += 1

        z = np.reshape(data, (NPKT*Nt*2,))
        z = z.astype(np.float64).view(np.complex128)

        y = np.reshape(z, (1, (NPKT*Nt)//Nfft, Nfft))
        Y = fft.fft(y, Nfft, axis=2)
        Ypsd = np.mean(np.real(Y*np.conj(Y)), axis=1)

        if line1 == None:
            line1, = ax.plot(faxis, 10*np.log10(fft.fftshift(Ypsd.transpose())))
        else:
            line1.set_ydata(10*np.log10(fft.fftshift(Ypsd.transpose())))
        ax.set_title('num: {:d}'.format(plt_cnt))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt_cnt += 1
