SRC = find_sinkholes

GDAL_CFLAGS = $(shell gdal-config --cflags)
GDAL_LIBDIR = $(shell gdal-config --prefix)/lib
GDAL_LIBS   = -L$(GDAL_LIBDIR) -lgdal -Wl,-rpath,$(GDAL_LIBDIR)
PDAL_CFLAGS = $(shell pdal-config --cflags)
PDAL_LIBDIR = $(shell pdal-config --prefix)/lib
PDAL_LIBS   = -L$(PDAL_LIBDIR) -lpdalcpp -Wl,-rpath,$(PDAL_LIBDIR)
CURL_LIBS   = -lcurl
CXXFLAGS_RELEASE = --std=c++17 -O3 -march=native -DNDEBUG -fpermissive -ffast-math -funroll-loops -flto=auto
CXXFLAGS_DEBUG   = --std=c++17 -O0 -g -fpermissive                                                                           
CXXFLAGS         = $(CXXFLAGS_RELEASE) $(GDAL_CFLAGS) -I$(SRC)      

OBJS = build/main.o build/dem.o build/utils.o build/FillDEM_Zhou-OnePass.o \
       build/find_sinkholes.o build/sinkhole.o build/settings.o build/colors.o \
       build/qgis_integration.o

all: bin/find_sinkholes

clean:                                                                                                                                                    
	rm -rf build
	rm -f bin/find_sinkholes 

bin/find_sinkholes: $(OBJS) | bin
	g++ $(OBJS) -flto=auto $(GDAL_LIBS) $(PDAL_LIBS) $(CURL_LIBS) -o $@

build/main.o: $(SRC)/main.cpp $(SRC)/settings.h $(SRC)/argparse.hpp $(SRC)/json.hpp | build
	g++ -c $< $(CXXFLAGS) $(PDAL_CFLAGS) -o $@

build/find_sinkholes.o: $(SRC)/find_sinkholes.cpp $(SRC)/dem.h $(SRC)/Cell.h $(SRC)/utils.h $(SRC)/settings.h $(SRC)/json.hpp $(SRC)/sinkhole.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/sinkhole.o: $(SRC)/sinkhole.cpp $(SRC)/sinkhole.h $(SRC)/utils.h $(SRC)/settings.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/settings.o: $(SRC)/settings.cpp $(SRC)/settings.h $(SRC)/colors.h $(SRC)/utils.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/dem.o: $(SRC)/dem.cpp $(SRC)/dem.h $(SRC)/utils.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/utils.o: $(SRC)/utils.cpp $(SRC)/utils.h $(SRC)/dem.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/FillDEM_Zhou-OnePass.o: $(SRC)/FillDEM_Zhou-OnePass.cpp $(SRC)/dem.h $(SRC)/Cell.h $(SRC)/utils.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/colors.o: $(SRC)/colors.cpp $(SRC)/colors.h | build
	g++ -c $< $(CXXFLAGS) -o $@

build/qgis_integration.o: $(SRC)/qgis_integration.cpp $(SRC)/qgis_integration.h $(SRC)/settings.h | build
	g++ -c $< $(CXXFLAGS) -o $@

bin build:
	mkdir -p $@

clean:
	rm -rf build bin
.PHONY: all clean
