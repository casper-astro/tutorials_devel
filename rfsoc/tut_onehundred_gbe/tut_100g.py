#!/home/mcb/opt/miniconda3/envs/casper-dev3/bin/python
import sys
import struct
import numpy as np

from multiprocessing.connection import Client
from socket import socket, AF_PACKET, SOCK_RAW, htons

SIZE_JUMBO_PKT = 9000
ETH_P_ALL = 0x3


Nbyte_per_time = 4  # bytes per sample (I/Q time sample)
Nbyte_hdr = 6
Nbyte_per_word = 64
Nword_per_pkt = 128

PAYLOAD_SIZE = Nword_per_pkt*Nbyte_per_word
PKT_SIZE = PAYLOAD_SIZE + Nbyte_hdr

Nt = PAYLOAD_SIZE//Nbyte_per_time  # time samples per packet

NPKT = 2**4

def catch(npkt):
  pkts = []
  i = 0
  while i < NPKT:
      pkts.append(s.recv(SIZE_JUMBO_PKT))
      i+=1
  return pkts

if __name__=="__main__":

    try:
        s = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))
        s.bind(("enp193s0f0", 0))
    except Exception:
        print("something went wrong opening ethernet interface...")

    try:
        address = ('localhost', 6000)
        conn = Client(address, authkey=b'secret password')
    except Exception as e:
        print("\n\nSomething went wrong connecting to listener socket...")
        print("Is the listener running? It must be started first.\n\n")
        print(e)
        sys.exit()


    print("sending capture parameters to listener...")
    conn.send([NPKT, PKT_SIZE, Nt])

    try:
        seq_cnt = 0
        while True:
            msg = None
            print("waiting for listener to be ready...", end='\r')
            while msg != "ready":
                msg = conn.recv()

            print("catching packets!", end='\r')
            p = catch(NPKT)

            print("sending packet sequence {:d}...".format(seq_cnt), end='\r')
            for i in range(0, NPKT):
                conn.send(p[i])
            seq_cnt+=1
    except KeyboardInterrupt:
        print("all done!")
        conn.send("close")

#print("catching packets...")
#pkts = catch(NPKT)
#print("done catching packets...")
#print("parsing packets...")
#
#mcnts = np.zeros((NPKT,1))
#d = np.zeros((NPKT, Nt))
#
#for i, p in enumerate(pkts):
#    eth_hdr = p[0:14]   # [dstmac, srcmac, ethtype]
#    ip_hdr  = p[14:34]  # [..., srcip, dstip]
#    udp_hdr = p[34:42]  # [src port, dst port, len]
#    pkt_hdr = p[42:106] # 64 byte alpaca header
#
#    #print(eth_hdr)
#    #print(ip_hdr)
#    #print(udp_hdr)
#
##    xid  = pkt_hdr[0:2]
##    fid  = pkt_hdr[2:4]
##    cal  = pkt_hdr[4:5]
##    mcnt = pkt_hdr[5:]
##
#    pkt_dat = p[106:]
#    #print(pkt_hdr)
#    pkt_cnt = struct.unpack("L", pkt_hdr[0:8])
#    mcnts[i]= pkt_cnt
#    #d[i,:] = struct.unpackt("4096h", pkt_dat)    
#    #d = struct.unpack("4096h", pkt_dat)
#    conn.send(p)
#
#conn.send("close")
