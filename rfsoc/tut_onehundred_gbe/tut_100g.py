#!/home/mcb/opt/miniconda3/envs/casper-dev3/bin/python
import sys, os, time
from multiprocessing.connection import Client
from socket import socket, AF_PACKET, SOCK_RAW, htons
import struct
import numpy as np


SIZE_JUMBO_PKT = 9000
ETH_P_ALL = 0x3


Nbyte_per_time = 4  # bytes per sample (I/Q time sample)
Nbyte_hdr = 6
Nbyte_per_word = 64
Nword_per_pkt = 128

PAYLOAD_SIZE = Nword_per_pkt*Nbyte_per_word
PKT_SIZE = PAYLOAD_SIZE + Nbyte_hdr

Nt = PAYLOAD_SIZE//Nbyte_per_time  # time samples per packet

print("Nt sampels:", Nt)

NPKT = 2**12

def catch(npkt):
  pkts = []
  i = 0
  while i < NPKT:
      pkts.append(s.recv(SIZE_JUMBO_PKT))
      i+=1
  return pkts

if __name__=="__main__":

    s = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))
    s.bind(("enp193s0f0", 0))


    address = ('localhost', 6000)
    conn = Client(address, authkey=b'secret password')

    print("sending info to listener...")
    conn.send([NPKT, PKT_SIZE, Nt])

    msg = None
    print("waiting for listener to be ready...")
    while msg != "ready":
        msg = conn.recv()
        print(msg)

    print("catching packets!")
    p = catch(NPKT)

    print("sending packets...")
    for i in range(0, NPKT):
        conn.send(p[i])

    print("all done!")


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
