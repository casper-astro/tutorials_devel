#!/usr/bin/python

# tut5.py
# CASPER Tutorial 5: Heterogeneous Instrumentation
#   Data acquisition script.

import socket, sys
import numpy as np

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('udp_grab.py <10GbE IP>')
    p.set_description(__doc__)
    opts, args = p.parse_args(sys.argv[1:])

    if args==[]:
        print 'Please specify 10GbE IP address. \nExiting.'
        exit()
    else:
        udp_ip = args[0]

    udp_port=60000
    size=1024

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((udp_ip, udp_port))

    for j in range(10):
        arr0=np.array([])
        filename='file'+'{0:04}'.format(j)
        for i in range(35000):
            data, addr = sock.recvfrom(size)
            arr = np.array(data)
            arr0 = np.append(arr0, arr)
        arr0.tofile(filename)

    sock.close()
 
