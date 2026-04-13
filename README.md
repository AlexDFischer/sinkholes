# Automatically extract sinkholes from LIDAR point cloud or DEM data

This tool takes in LIDAR point cloud data, or digital elevation models (DEMs), and extracts sinkholes from this elevation data.
It is primarily intended for use by cavers, to find caves. Closed depressions may be cave entrances, and systematically checking closed depressions in areas with the right geology to have caves is a good way to find caves.

This program automatically extracts sinkholes from DEMs using depression filling algorithms (specifically a modified version of [fast priority-flood](https://www.sciencedirect.com/science/article/abs/pii/S0098300416300553)). It can also input point cloud .las/.laz files and turn them into DEMs (using PDAL). You don't even have to give the program point cloud files: you can specify the coordinates of a bounding box that specify an area you're interested in in the US, and it will automatically download point clouds from the [USGS's data download service](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download).

The program has two primary outputs: hillshade maps with sinkholes highlighted and colorcoded by depth, and .geojson files listing sinkholes with various statistics such as depth, area, and elevation. The program can automatically add these to a QGIS project you specify, and style layers how you like.

![One sinkhole highlighted by depth on a hillshade map](docs/sinkhole_hillshade_screenshot.png)

One sinkhole highlighted by depth on a hillshade map.

![Many sinkholes labelled on a hillshade map](docs/sinkholes_screenshot.png)

Many sinkholes labelled on a hillshade map.

# Installation

Right now you have to build from source. The project is written in C++ and requires a C++ compiler. Clone the repository, navigate to the root directory of the repository, and run `make`. The Makefile uses `g++`; it should compile with any other C++ compiler if you change the Makefile.

To use the QGIS integration feature, you'll have to have [QGIS](https://qgis.org/) installed. Download from the [QGIS website](https://qgis.org/). This program still works without QGIS installed; you just won't be able to use the QGIS integration feature.

You shouldn't have to install any Python beyond what comes installed automatically with QGIS. This program will automatically find and use the Python installation that comes with your QGIS.

# Usage

The `find_sinkholes` program in `bin/` is the executable. Run it with the following arguments:

 * `-ll`, `--lower-left`. Lower left coordinate of boudning box around your region of interest. LIDAR point clouds from the USGS in this area will be automatically downloaded.
 * `-ur`, `--upper-right`. Upper right coordinate of boudning box around your region of interest. LIDAR point clouds from the USGS in this area will be automatically downloaded.
 * `-pc`, `--point-clouds`. Point cloud files in .las/laz format. Use this option instead of the above two flags if you already have the point cloud data you want to use, and don't need to download it from the USGS.
 * `-d`, `--dem`. Input digital elevation models (DEMs) to use. Use this option instead of any of the above input options if you already have the DEMs you want to find sinkholes in, and you don't need to create any from point clouds.

All of the above input flags are optional, but you need to specify at least one input file. I recommend using the `-ll` and `-ur` options to download point cloud data that will then be automatically turned into DEMs from which sinkholes can be extracted.

 * `-oh`, `--output_hillshade`. Output directory (or .tif file, if there is only one input file) for generated hillshade maps. 
 * `-os`, `--output_sinkholes`. Output directory (or .geojson file, if there is only one input file) for generated sinkhole point lists.

You must specify at least one of the above output options. Below are some optional args.

 * `-q`, `--qgis`. QGIS project file (.qgz or .qgs) to add generated hillshade maps and sinkholes lists to.
 * `-s`, `--settings`. Settings file. The `settings.json` has all the optional settings and is in the requireed format.

## Recommended usage for beginners

First, make a copy of the `qgis_template` folder somewhere and rename `template.qgz` to your liking. Then pick an area that you're interested in finding caves in. I recommend it be no more than about 100 km^2. Get the lower-left and upper-right coordinates of a bounding box around this area. Run the following command:

`bin/find_sinkholes -ll <lower_right_coordinate_of_bounding_box> -ur <upper_right_coordinate_of_bounding_box> -oh <folder_with_qgis_project> -oh <folder_with_qgis_project> --qgis <folder_with_qgis_project>/<qgis_project_filename.qgz>`

For those outside the US, I don't know what data is generally available. Use point cloud files or DEMs from wherever you can get them.

## A note on colormaps

The default colormap is rainbow_4 (reversed) from [colorcet](https://colorcet.com/gallery.html#rainbow). Rainbow colormaps are not [perceptually uniform](https://programmingdesignsystems.com/color/perceptually-uniform-color-spaces/) and are often not recommended for visualizing 3D objects like how they are being used here. However, I have chose to use a rainbow colormap anyways, because I care less about the perceptual uniformity of the sinkhole depth visualizations, and more about the visual contrast against the grey background and the range of depth values that can be distinguished. Rainbow colormaps are superior to standard perceptualy uniform colormaps at those tasks. See below.

# Included QGIS template

I have included a QGIS project in the `qgis_template` directory. This project file has several layers one will find useful for cave-hunting: 