import sys
import struct
import numpy as np

from multiprocessing.connection import Client
from socket import socket, AF_PACKET, SOCK_RAW, htons

SIZE_JUMBO_PKT = 9000
ETH_P_ALL = 0x3

def catch_packets(npkt):
  pkts = []
  i = 0
  while i < NPKT:
      pkts.append(s.recv(SIZE_JUMBO_PKT))
      i+=1
  return pkts

if __name__=="__main__":
    # open ethernet interface for catching raw ethernet packets
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('rfsoc4x2_tut_100g_catcher.py <ethernet_interface>')
    p.set_description(__doc__)

    opts, args = p.parse_args(sys.argv[1:])
    if len(args) < 1:
      print('Must specify an ethernet interface name to open. Run with the -h flag to see all options.\n')
      sys.exit()
    else:
      raw_sock_intf = args[0]

    try:
        print("opening ethernet interface {:s}".format(raw_sock_intf))
        s = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))
        s.bind((raw_sock_intf, 0))
    except Exception:
        print("\n\nSomething went wrong opening ethernet interface...")
        print("Is this a valid ethernet interface name?\n\n")
        print(e)
        sys.exit()

    # open and connect to listener socket for passing off data for plotting
    try:
        address = ('localhost', 6000)
        conn = Client(address, authkey=b'secret password')
    except Exception as e:
        print("\n\nSomething went wrong connecting to listener socket...")
        print("Is the listener running? It must be started first.\n\n")
        print(e)
        sys.exit()

    # receive configuration parameters from listener socket
    param = conn.recv()
    NPKT = param[0]
    PKT_SIZE = param[1]
    Nt   = param[2]
    print("configured to catching {:d} packets each with {:d} time samples".format(NPKT, Nt))

    print("\n\nTerminte with keyboard interrupt here, this will also close the listener")
    # begin catching packets
    try:
        seq_cnt = 0
        while True:
            msg = None
            #print("waiting for listener to be ready...", end='\r')
            while msg != "ready":
                msg = conn.recv()

            #print("catching packets!", end='\r')
            p = catch_packets(NPKT)

            print("sending packet sequence {:d}...".format(seq_cnt), end='\r')
            for i in range(0, NPKT):
                conn.send(p[i])
            seq_cnt+=1
    except KeyboardInterrupt:
        print("all done!")
        conn.send("close")
