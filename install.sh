# carla simulator 
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
`sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"`
sudo apt-get update *# Update the Debian package index*
sudo apt-get install carla-simulator *# Install the latest CARLA version, or update the current installation*

# install requirements
pip install -r requirements.txt
