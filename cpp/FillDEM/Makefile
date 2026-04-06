GDAL_CFLAGS = $(shell gdal-config --cflags)
GDAL_LIBDIR = $(shell gdal-config --prefix)/lib
GDAL_LIBS = -L$(GDAL_LIBDIR) -lgdal -Wl,-rpath,$(GDAL_LIBDIR)
CXXFLAGS = --std=c++17 -fpermissive $(GDAL_CFLAGS)

find_sinkholes: dem.o utils.o FillDEM_Zhou-OnePass.o find_sinkholes.o
	g++ dem.o utils.o FillDEM_Zhou-OnePass.o find_sinkholes.o \
	$(GDAL_LIBS) -o bin/find_sinkholes

fillDEM: dem.o main.o utils.o FillDEM_Zhou-Direct.o FillDEM_Zhou-OnePass.o FillDEM_Zhou-TwoPass.o
	g++ dem.o main.o utils.o FillDEM_Zhou-Direct.o FillDEM_Zhou-OnePass.o FillDEM_Zhou-TwoPass.o \
	$(GDAL_LIBS) -o bin/fillDEM

bug: bug.o dem.o utils.o
	g++ bug.o dem.o utils.o $(GDAL_LIBS) -o bin/bug

bug.o: dem.h Node.h utils.h bug.cpp
	g++ -c bug.cpp $(CXXFLAGS) -o bin/bug.o

find_sinkholes.o: dem.h Node.h utils.h settings.h argparse.hpp FillDEM_Zhou-OnePass.cpp settings.cpp find_sinkholes.cpp
	g++ -c find_sinkholes.cpp $(CXXFLAGS) -o bin/find_sinkholes.o

dem.o: dem.h utils.h dem.cpp
	g++ -c dem.cpp $(CXXFLAGS) -o bin/dem.o
main.o: dem.h Node.h utils.h FillDEM_Zhou-Direct.cpp FillDEM_Zhou-OnePass.cpp FillDEM_Zhou-TwoPass.cpp main.cpp
	g++ -c main.cpp $(CXXFLAGS) -o bin/main.o
utils.o: dem.h utils.h utils.cpp
	g++ -c utils.cpp $(CXXFLAGS) -o bin/utils.o
FillDEM_Zhou-Direct.o: dem.h Node.h utils.h FillDEM_Zhou-Direct.cpp
	g++ -c FillDEM_Zhou-Direct.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-Direct.o
FillDEM_Zhou-OnePass.o: dem.h Node.h utils.h FillDEM_Zhou-OnePass.cpp
	g++ -c FillDEM_Zhou-OnePass.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-OnePass.o
FillDEM_Zhou-TwoPass.o: dem.h Node.h utils.h FillDEM_Zhou-TwoPass.cpp
	g++ -c FillDEM_Zhou-TwoPass.cpp $(CXXFLAGS) -o bin/FillDEM_Zhou-TwoPass.o

clean:
	@echo "cleanning project"
	-rm -rf bin/*
	@echo "clean completed"
.PHONY: clean

