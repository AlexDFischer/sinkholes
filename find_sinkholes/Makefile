GDAL_CFLAGS = $(shell gdal-config --cflags)
GDAL_LIBDIR = $(shell gdal-config --prefix)/lib
GDAL_LIBS = -L$(GDAL_LIBDIR) -lgdal -Wl,-rpath,$(GDAL_LIBDIR)
PDAL_CFLAGS = $(shell pdal-config --cflags)
PDAL_LIBDIR = $(shell pdal-config --prefix)/lib
PDAL_LIBS = -L$(PDAL_LIBDIR) -lpdalcpp -Wl,-rpath,$(PDAL_LIBDIR)
CURL_LIBS = -lcurl
CXXFLAGS = --std=c++17 -fpermissive $(GDAL_CFLAGS)

all: find_sinkholes

find_sinkholes: bin/main.o bin/dem.o bin/utils.o bin/FillDEM_Zhou-OnePass.o bin/find_sinkholes.o bin/sinkhole.o bin/settings.o bin/colors.o
	g++ bin/main.o bin/dem.o bin/utils.o bin/FillDEM_Zhou-OnePass.o bin/find_sinkholes.o bin/sinkhole.o bin/settings.o bin/colors.o \
	$(GDAL_LIBS) $(PDAL_LIBS) $(CURL_LIBS) -o bin/find_sinkholes

bin/main.o: main.cpp settings.h argparse.hpp json.hpp
	g++ -c main.cpp $(CXXFLAGS) $(PDAL_CFLAGS) -o bin/main.o

bin/colors.o: colors.h colors.cpp
	g++ -c colors.cpp $(CXXFLAGS) -o bin/colors.o

bin/find_sinkholes.o: dem.h Cell.h utils.h settings.h json.hpp sinkhole.h find_sinkholes.cpp
	g++ -c find_sinkholes.cpp $(CXXFLAGS) -o bin/find_sinkholes.o

bin/sinkhole.o: utils.h settings.h sinkhole.h sinkhole.cpp
	g++ -c sinkhole.cpp $(CXXFLAGS) -o bin/sinkhole.o

bin/settings.o: utils.h colors.h settings.h settings.cpp
	g++ -c settings.cpp $(CXXFLAGS) -o bin/settings.o

bin/dem.o: dem.h utils.h dem.cpp
	g++ -c dem.cpp $(CXXFLAGS) -o bin/dem.o

bin/utils.o: dem.h utils.h utils.cpp
	g++ -c utils.cpp $(CXXFLAGS) -o bin/utils.o
	
bin/FillDEM_Zhou-OnePass.o: dem.h Cell.h utils.h FillDEM_Zhou-OnePass.cpp
	g++ -c FillDEM_Zhou-OnePass.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-OnePass.o

clean:
	@echo "cleanning project"
	-rm -rf bin/*
	@echo "clean completed"
.PHONY: clean

