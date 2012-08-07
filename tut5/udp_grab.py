#!/opt/vegas/bin/python

import socket
import numpy as np

udp_ip='10.0.1.146'
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
 
