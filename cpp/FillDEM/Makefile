GDAL_CFLAGS = $(shell gdal-config --cflags)
GDAL_LIBDIR = $(shell gdal-config --prefix)/lib
GDAL_LIBS = -L$(GDAL_LIBDIR) -lgdal -Wl,-rpath,$(GDAL_LIBDIR)
CXXFLAGS = --std=c++17 -fpermissive $(GDAL_CFLAGS)

find_sinkholes: bin/dem.o bin/utils.o bin/FillDEM_Zhou-OnePass.o bin/find_sinkholes.o bin/sinkhole.o bin/settings.o bin/colors.o
	g++ bin/dem.o bin/utils.o bin/FillDEM_Zhou-OnePass.o bin/find_sinkholes.o bin/sinkhole.o bin/settings.o bin/colors.o \
	$(GDAL_LIBS) -o bin/find_sinkholes

fillDEM: bin/dem.o bin/main.o bin/utils.o bin/FillDEM_Zhou-Direct.o bin/FillDEM_Zhou-OnePass.o bin/FillDEM_Zhou-TwoPass.o
	g++ bin/dem.o bin/main.o bin/utils.o bin/FillDEM_Zhou-Direct.o bin/FillDEM_Zhou-OnePass.o bin/FillDEM_Zhou-TwoPass.o \
	$(GDAL_LIBS) -o bin/fillDEM

bug: bin/bug.o bin/dem.o bin/utils.o
	g++ bin/bug.o bin/dem.o bin/utils.o $(GDAL_LIBS) -o bin/bug

bin/bug.o: dem.h Cell.h utils.h bug.cpp
	g++ -c bug.cpp $(CXXFLAGS) -o bin/bug.o

bin/colors.o: colors.h colors.cpp
	g++ -c colors.cpp $(CXXFLAGS) -o bin/colors.o

bin/find_sinkholes.o: dem.h Cell.h utils.h settings.h argparse.hpp json.hpp settings.h sinkhole.h find_sinkholes.cpp
	g++ -c find_sinkholes.cpp $(CXXFLAGS) -o bin/find_sinkholes.o

bin/sinkhole.o: utils.h settings.h sinkhole.h sinkhole.cpp
	g++ -c sinkhole.cpp $(CXXFLAGS) -o bin/sinkhole.o

bin/settings.o: utils.h colors.h settings.h settings.cpp
	g++ -c settings.cpp $(CXXFLAGS) -o bin/settings.o

bin/dem.o: dem.h utils.h dem.cpp
	g++ -c dem.cpp $(CXXFLAGS) -o bin/dem.o
bin/main.o: dem.h Cell.h utils.h FillDEM_Zhou-Direct.cpp FillDEM_Zhou-OnePass.cpp FillDEM_Zhou-TwoPass.cpp main.cpp
	g++ -c main.cpp $(CXXFLAGS) -o bin/main.o
bin/utils.o: dem.h utils.h utils.cpp
	g++ -c utils.cpp $(CXXFLAGS) -o bin/utils.o
bin/FillDEM_Zhou-Direct.o: dem.h Cell.h utils.h FillDEM_Zhou-Direct.cpp
	g++ -c FillDEM_Zhou-Direct.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-Direct.o
bin/FillDEM_Zhou-OnePass.o: dem.h Cell.h utils.h FillDEM_Zhou-OnePass.cpp
	g++ -c FillDEM_Zhou-OnePass.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-OnePass.o
bin/FillDEM_Zhou-TwoPass.o: dem.h Cell.h utils.h FillDEM_Zhou-TwoPass.cpp
	g++ -c FillDEM_Zhou-TwoPass.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-TwoPass.o

clean:
	@echo "cleanning project"
	-rm -rf bin/*
	@echo "clean completed"
.PHONY: clean

