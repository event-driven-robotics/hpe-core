
#!/bin/bash

# install yarp
sudo sh -c 'echo "deb http://www.icub.org/ubuntu focal contrib/science" > /etc/apt/sources.list.d/icub.list'
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 57A5ACB6110576A6
sudo apt-get update
sudo apt-get install -y yarp

# make yarpserver in docker container visible to local yarp
yarp conf 172.17.0.2 10000
yarp check
yarp detect

# launch yarpview
yarpview --name /img_vis --x 30 --y 30 --h 720 --w 960 --synch --compact
