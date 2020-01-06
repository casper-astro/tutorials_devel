# Guide to Setting Up Your New Red Pitaya

This guide will show you how to setup your Red Pitaya(RP) for use with the casper tools - mlib_devel and CASPERFPGA.

* How to create the ISO for the RP SD card,
* How to setup tcpborphserver on the RP,

You have 2 options when it comes to setting up the SD card.

* Running a script on the sd card you were supplied with the RP with the OS pre installed
* or, creating the SD card using a casper supplied image. (I have not found somewhere to host this file yet so please use the above method for now)


## Running the script on a preloaded RP SD Card

- This will not affect the running of the native RP software, they run happily side-by-side 
- Insert your SD card and boot the RP
- SSH into the RP using the hostname printed on the Ethernet port of the board (default user:root password:root)
- Run the following script on the RP:

```bash
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

- This will install git, clone and build tcpborphserver and then set it to run on startup.
- Your RP is now casperized. You can communicate with it via CASPERFPGA running on a remote server.

- If you need to install a blank SD with the RP OS please follow the setup instructions on the RP site [here](https://redpitaya.readthedocs.io/en/latest/quickStart/SDcard/SDcard.html)

## Download a casperized SD card image.

- This is not yet available, hopefully coming soon. 
