sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt install libgdal-dev

export CPLUS_INCLUDE_PATH=/usr/include/gdal

export C_INCLUDE_PATH=/usr/include/gdal

pip install -r requirements.txt