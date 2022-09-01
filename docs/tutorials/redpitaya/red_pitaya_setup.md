# Guide to Setting Up Your New Red Pitaya

This guide will show you how to setup your Red Pitaya (RP) for use with the casper tools - `mlib_devel` and `casperfpga`. 

Setting up the Red Pitaya involves installing and building a few things through script commands on the Red Pitaya OS. If you need to install a blank SD card with the Red Pitaya OS please follow the setup instructions on the Red Pitaya site [here](https://redpitaya.readthedocs.io/en/latest/quickStart/SDcard/SDcard.html), otherwise use the SD card that came supplied with the hardware.


## Running the script on a preloaded RP SD Card

- This will not affect the running of the native RP software, they run happily side-by-side 
- Insert your SD card and boot the RP, ensuring that the RP is connected to your Ethernet network
- SSH into the RP using the hostname printed on the Ethernet port of the board (default user:root password:root)
- Run the following script on the RP:

```bash
# make sure /etc/hosts exists
touch /etc/hosts
# make sure localhost is in /etc/hosts (required by tcpborphserver3)
grep -q localhost /etc/hosts || echo "127.0.0.1 localhost" >> /etc/hosts
# install git
apt-get install git
# clone katcp
git clone https://github.com/ska-sa/katcp.git
# build katcp
cd katcp
make all
# copy executables to /bin
cp cmd/kcpcmd /bin/
cp fpg/kcpfpg /bin/
cp tcpborphserver3/tcpborphserver3 /bin/
# create startup service file
echo "Description=TCPBorphServer allows programming and communication with the FPGA
Wants=network.target
After=syslog.target network-online.target
[Service]
Type=simple
ExecStart=/bin/tcpborphserver3
Restart=on-failure
RestartSec=10
KillMode=process
[Install]
WantedBy=multi-user.target" > /etc/systemd/system/tcpborphserver.service
# reload services
systemctl daemon-reload
# enable the service
systemctl enable tcpborphserver
# start the service
systemctl start tcpborphserver
# check the status of your service
systemctl status tcpborphserver

```

- This will install git on the RP, clone and build tcpborphserver (a server designed to control and speak to CASPER hardware using KATCP) and then set it to run on startup.
- Your RP is now casperized and you can communicate with it via `casperfpga`. For details on installing `casperfpga`, please see [here](https://casper-toolflow.readthedocs.io/en/latest/src/How-to-install-casperfpga.html).


