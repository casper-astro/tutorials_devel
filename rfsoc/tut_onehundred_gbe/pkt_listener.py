#!/home/mcb/opt/miniconda3/envs/casper-dev3/bin/python
from multiprocessing.connection import Listener
import struct
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

#NPKT = 2**4
#Nt = 2048
#d = np.zeros((NPKT,4096), dtype=int)

#pkts = []
#i = 0
#while i < NPKT:
#  recv_d = conn.recv()
#  if recv_d == 'close':
#      conn.close()
#      break
#  #d[i,:] = recv_d.astype(np.float64).view(np.complex128)
#  #d[i,:] = recv_d
#  pkts.append(recv_d)
#  # do something with msg
#
#listener.close()
#
#Nfft = 2**10


#    eth_hdr = p[0:14]   # [dstmac, srcmac, ethtype]
#    ip_hdr  = p[14:34]  # [..., srcip, dstip]
#    udp_hdr = p[34:42]  # [src port, dst port, len]
#    pkt_hdr = p[42:106] # 64 byte alpaca header
#
#    #print(eth_hdr)
#    #print(ip_hdr)
#    #print(udp_hdr)
#
if __name__ == "__main__":

    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)

    d = conn.recv()
    NPKT = d[0]
    PKT_SIZE = d[1]
    Nt   = d[2]
    payload_fmt_str = "{:d}h".format(Nt*2)
    print("catching {:d} packets each with {:d} time samples".format(NPKT, Nt))

    # setup from parameters before continuing
    pkts = []

    conn.send('ready')

    i = 0
    while i < NPKT:
        r = conn.recv()
        pkts.append(r)
        i+=1

    listener.close()

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
        vld += 1

    z = np.reshape(data, (NPKT*Nt*2,))
    z = z.astype(np.float64).view(np.complex128)

    Nfft = 2**10
    fbins = np.arange(-Nfft//2, Nfft//2)
    fs = 3932.16/2
    df = fs/Nfft
    faxis = df*fbins + fs/2

    y = np.reshape(z, (1, (NPKT*Nt)//Nfft, Nfft))
    Y = fft.fft(y, Nfft, axis=2)
    Ypsd = np.mean(np.real(Y*np.conj(Y)), axis=1)

    plt.plot(faxis, 10*np.log10(fft.fftshift(Ypsd.transpose()))); plt.grid();
    plt.show();

