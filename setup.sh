wget https://deb.debian.org/debian/pool/main/g/gdal/gdal_3.9.0+dfsg-1.dsc

sudo dpkg -i gdal_3.9.0+dfsg-1.dsc

export CPLUS_INCLUDE_PATH=/usr/include/gdal

export C_INCLUDE_PATH=/usr/include/gdal

pip install -r requirements.txt