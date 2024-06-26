#!/bin/python3

from datetime import datetime
import math
import numpy as np
import earthpy.spatial as es
import gc
import pyproj
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
import richdem as rd
import cv2
import json
import time
import argparse
import traceback
import uuid

from sinkhole import Sinkhole
from util import feet_per_meter, gaia_datetime_format
from color_utils import ColorUtil

wgs84_name = 'EPSG:4326'

def process_geotiff(geotiff_input_filename, geotiff_output_filename, sinkholes_output_filename,
                    config,
                    output_geotiff=True,
                    output_geojson=True):
    
    geotiff_input = rasterio.open(geotiff_input_filename, nodata=0)
    elevation = geotiff_input.read(1)
    elevation[elevation<0] = 0
    color_util = ColorUtil(config['min_depth_for_colormap'], config['max_depth_for_colormap'], config['pin_colormap'], config['map_colormap'])

    print('Loaded geotiff DEM.')

    # fill depressions, get difference, and get difference
    rich_dem = rd.rdarray(elevation, no_data=0)
    diff = np.array(rd.FillDepressions(rich_dem) - rich_dem)
    del rich_dem # save memory
    gc.collect()

    print('Done with depression filling.')

    if output_geotiff:
        # make hillshade map with sinkholes highlighted by depth
        hillshade = es.hillshade(elevation, azimuth=315, altitude=30)
        img = np.zeros(shape=(3, hillshade.shape[0], hillshade.shape[1]), dtype=np.uint8)
        for channel in range(3):
            img[channel, :, :] = hillshade # img has all 3 channels equal to hillshade map (so img is black and white hillshade map)

        del hillshade # save some memory
        gc.collect()

        map_color_ufunc = np.frompyfunc(color_util.depth_to_map_color, 1, 1)
        nonzero_diff_index = diff > 0
        diff_colors = np.stack(map_color_ufunc(diff[nonzero_diff_index])) # need to stack because numpy wants to store the output of map_color_ufunc as an array of arrays
        diff_colors = np.moveaxis(diff_colors, -1, 0) # rearrange axes so last index is index for 3 color channels
        img[:, nonzero_diff_index] = diff_colors
  
        del nonzero_diff_index # save some memory
        gc.collect()

        output_profile = geotiff_input.profile
        output_profile.update(dtype=rasterio.uint8, count=3, nodata=0)
        with rasterio.open(geotiff_output_filename, 'w', **output_profile) as geotiff_output:
            geotiff_output.write(img)
        
        print('Exported geotiff map.')

    if output_geojson:
        print('Depth to pin color mapping for GaiaGPS:')
        print(color_util.gaia_colormap_string(config['units']))

        time_before_sinkholes = time.time()
        sinkholes = sinkholes_from_diff(diff, geotiff_input, elevation, config['min_depth'], config['max_dimension'])
        time_after_sinkholes = time.time()
        print(f'Found {len(sinkholes)} sinkholes. Elapsed time making sinkhole objects: {time_after_sinkholes - time_before_sinkholes:.2f} s.')
        export_sinkholes_geojson(sinkholes, sinkholes_output_filename, color_util, units=config['units'], max_points_per_file=config['max_points_per_file'])
        print('Exported sinkhole objects to geojson file(s).')
    

def sinkholes_from_diff(diff, geotiff_input, elevation, min_depth, max_dimension):
    """max dimension is the maximum width or length allowed for a sinkhole before we no longer include it"""

    # make wgs84_transformer object, that transforms from the built in coordinate reference system of the geotiff we're reading
    wgs84_transformer = pyproj.Transformer.from_crs(pyproj.CRS(str(geotiff_input.crs)), pyproj.CRS(wgs84_name))

    diffs_nonzero = ((diff >= min_depth) * 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diffs_nonzero, connectivity=4, ltype=cv2.CV_16U)

    sinkholes = []

    max_depths = np.zeros((num_labels))
    max_depth_locations = np.zeros((num_labels, 2))
    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            depth = diff[row, col]
            label = labels[row, col]
            if depth > max_depths[label]:
                max_depths[label] = depth
                max_depth_locations[label] = [row, col]
    
    for label in range(num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        length = stats[label, cv2.CC_STAT_HEIGHT]
        if width <= max_dimension and length <= max_dimension:
            row = max_depth_locations[label, 0]
            col = max_depth_locations[label, 1]
            input_crs_coords = geotiff_input.transform * (col, row)
            lat, long = wgs84_transformer.transform(input_crs_coords[0], input_crs_coords[1])
            row = int(round(row))
            col = int(round(col))

            sinkhole = Sinkhole(depth=max_depths[label],
                                lat=lat,
                                long=long,
                                width=width,
                                length=length,
                                elevation=elevation[row, col]+diff[row, col],
                                area=stats[label, cv2.CC_STAT_AREA])
            sinkholes.append(sinkhole)

    return sinkholes

def export_sinkholes_geojson(sinkholes, output_filename, color_util, units='metric', max_points_per_file=-1):
    if output_filename == None or output_filename == '':
        raise ValueError('Must specify an output filename')
    
    folder_uuid = uuid.uuid4()

    if max_points_per_file > 0 and len(sinkholes) > max_points_per_file:
        # split up the sinkholes into multiple files
        fname_arr = output_filename.split('.')
        fname_prefix = '' # part of filename before extension
        fname_extension = ''
        if len(fname_arr) == 1:
            # there's no file extension
            fname_prefix = output_filename
        else:
            fname_prefix = '.'.join(fname_arr[:-1])
            fname_extension = '.' + fname_arr[-1]

        num_files = (len(sinkholes) + max_points_per_file - 1) // max_points_per_file
        num_decimal_digits = math.ceil(math.log10(num_files))
        for i in range(num_files):
            new_output_fname = fname_prefix + ('_{:0' + str(num_decimal_digits) + 'n}').format(i) + fname_extension
            export_sinkholes_geojson(sinkholes[i * max_points_per_file : (i+1) * max_points_per_file], new_output_fname, color_util, units)
    else:
        unit_conversion = None
        unit_str = None
        if units == 'metric':
            unit_conversion = 1.0
            unit_str = 'm'
        elif units == 'imperial':
            unit_conversion = feet_per_meter
            unit_str = 'ft'
        else:
            raise ValueError(f'Error: `"units`" was \"{units}\", but the only allowed values are \"metric\" or \"imperial\".')
        
        now = datetime.now()
        min_depth = min([sinkhole.depth for sinkhole in sinkholes])
        max_depth = max([sinkhole.depth for sinkhole in sinkholes])
        output = {
            "type": "FeatureCollection", # folder info for GaiaGPS
            "properties": {
                "name": "caves.science automatic sinkholes",
                "updated_date": now.strftime(gaia_datetime_format),
                "time_created": now.strftime(gaia_datetime_format),
                "notes": f"""{len(sinkholes)} sinkholes automatically detected by caves.science.
                                Depths range from {min_depth * unit_conversion}:.1f {unit_str} to {max_depth * unit_conversion:.1f} {unit_str}."""
            },
            "features": [{ # folder info for Caltopo
                "geometry": None,
                "id": folder_uuid.hex,
                "type": "Feature",
                "properties":{
                    "visible": True,
                    "title": "caves.science automatic sinkholes",
                    "class": "Folder",
                    "labelVisible": True
                }
            }] + [sinkhole.json_obj(color_util, folder_uuid, units=units) for sinkhole in sinkholes]
        }

        file = open(output_filename, 'w')
        file.write(json.dumps(output, indent=4))

def default_config():
    return {
        'min_depth': 0.5, # minimum depth for a sinkhole to be counted
        'max_dimension': 300, # sinkholes with either E-W size or N-S size larger than this are not counted
        'min_depth_for_colormap': 0.5, # colormap for depth starts at this value. Sinkholes shallower than this depth get color that indicates min depth
        'max_depth_for_colormap': 6, # colormap for depth ends at this value. Sinkholes deeper than this depth get color that indicates max depth
        'max_points_per_file': -1, # if there are more than this number of points, split them up into multiple files (useful because e.g. gaiagps can't handle more than 1000 points per file). Set it to -1 for no max
        "pin_colormap": "gist_rainbow", # matplotlib colormap name to use for pin color. Not recommended to be the same as map_depth_colormap because pin_color_colormap will be used with a log scale, unlike map_depth_colormap
        "map_colormap": "inferno_r", # matplotlib colormap name to use for depth colorcoding in output map. Not recommended to be the same as pin_color_colormap because pin_color_colormap will be used with a log scale, unlike map_depth_colormap
        "units": "metric",
        "verbose": False
    }

parser = argparse.ArgumentParser(prog="Find Sinkholes", description="Automatically find sinkholes using 1m DEMs from USGS")
parser.add_argument('-i', '--input', action='store')
parser.add_argument('-otif', '--output-geotiff')
parser.add_argument('-ojson', '--output-geojson')
parser.add_argument('-c', '--config')
args = parser.parse_args()

if not ('input' in args and args.input != None):
    print('Must have --input (or -i) argument that specifies input .tif file with DEM')
    exit(1)

config = default_config()
if 'config' in args and args.config != None:
    try:
        with open(args.config, 'r') as config_file:
            config_json = json.loads(config_file.read())
            if 'min_depth' in config_json and type(config_json['min_depth']) in (int, float):
                config['min_depth'] = config_json['min_depth']
            if 'max_dimension' in config_json and type(config_json['max_dimension']) in (int, float):
                config['max_dimension'] = config_json['max_dimension']
            if 'min_depth_for_colormap' in config_json and type(config_json['min_depth_for_colormap']) in (int, float):
                config['min_depth_for_colormap'] = config_json['min_depth_for_colormap']
            if 'max_depth_for_colormap' in config_json and type(config_json['max_depth_for_colormap']) in (int, float):
                config['max_depth_for_colormap'] = config_json['max_depth_for_colormap']
            if 'max_points_per_file' in config_json and type(config_json['max_points_per_file']) in (int, float):
                config['max_points_per_file'] = config_json['max_points_per_file']
            if 'pin_colormap' in config_json and type(config_json['pin_colormap']) is str:
                config['pin_colormap'] = config_json['pin_colormap']
            if 'map_colormap' in config_json and type(config_json['map_colormap']) is str:
                config['map_colormap'] = config_json['map_colormap']
            if 'units' in config_json and type(config_json['units']) in ('metric', 'imperial'):
                config['units'] = config_json['units']
            if 'verbose' in config_json and type(config_json['verbose']) is bool:
                config['verbose'] = config_json['verbose']
    except FileNotFoundError:
        print('No config.json file found. Using default values.')
        config = default_config()
    except Exception as err:
        print('Error while readong config.json file.')
        print(traceback.format_exc())
        print('Using default config values')
        config = default_config()


input = args.input

output_geotiff = 'output_geotiff' in args and args.output_geotiff != None
output_geojson = 'output_geojson' in args and args.output_geojson != None

output_geotiff_fname = ''
output_geojson_fname = ''
if output_geotiff:
    output_geotiff_fname = args.output_geotiff
if output_geojson:
    output_geojson_fname = args.output_geojson

if config['verbose']:
    print('Using config object:')
    print(json.dumps(config, indent=4))

process_geotiff(input, output_geotiff_fname, output_geojson_fname,
                config=config,
                output_geotiff=output_geotiff,
                output_geojson=output_geojson)