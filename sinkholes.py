#!/bin/python3

import cv2
from datetime import datetime
import math
import numpy as np
import earthpy.spatial as es
import gc
import json
import pyproj
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import richdem as rd
import tempfile
import time
import argparse
import traceback
import uuid
from pathlib import Path
import requests
import wget
import os

from color_utils import ColorUtil
from config import default_config, parse_config
from sinkhole import Sinkhole
from util import feet_per_meter, meters_per_foot, gaia_datetime_format, wgs84_epsg, wgs84_name

def download_dems(bounding_box, data_dir=None):
    """Downloads 1m geoTIFF files in a given area from USGS servers.
    Args:
        bounding_box (str): two corners of rectangle defining search area in format 'lat1, lon1, lat2, lon2'
        data_dir (Path): directory in which to store downloaded files. Defaults to ./inputDEMs/
    Returns:
        filenames (list(str)): string representation of paths to downloaded files"""
    
    # reformat the bounding box to match the query syntax
    y1, x1, y2, x2 = list(map(float, bounding_box.split(','))) 
    ulx = min(x1, x2)
    uly = max(y1, y2)
    lrx = max(x1, x2)
    lry = min(y1, y2)
    bounding_box = f"{ulx}, {uly}, {lrx}, {lry}"
    if data_dir is None:
        data_dir = Path("./inputDEMs/")
    data_dir.mkdir(exist_ok=True)

    opr_dataset = "Original%20Product%20Resolution%20(OPR)%20Digital%20Elevation%20Model%20(DEM)"
    dem_1m_dataset = "Digital%20Elevation%20Model%20(DEM)%201%20meter"
    for dataset in [dem_1m_dataset]: # can also set to [opr_dataset, dem_1m_dataset] if higher res desired
        url = f"https://tnmaccess.nationalmap.gov/api/v1/products?&datasets={dataset}&bbox={bounding_box}&prodFormats=&max=1000&offset=0"
        
        try:
            result = requests.get(url=url, timeout=10)
        except requests.exceptions.ReadTimeout:
            print(f"{url} failed to elicit a reply")
            raise
        results = [(item['title'], item['downloadURL']) for item in json.loads(result.text)["items"]]
        if len(results) > 0: break

    if results: results = [results[0]]  #TODO: for now, just download the first result. In the long run, join all results and crop to bounding box
    
    # check total download size to make sure we aren't accidentally overloading the servers
    tot_size = 0
    MAX_DL_SIZE = 500 # in megabytes
    for result in results:
        response = requests.head(result[1], allow_redirects=True)
        size = int(response.headers.get('content-length', -1), )
        tot_size += size
    tot_size = tot_size / float(1 << 20) # convert to MB
    if tot_size > MAX_DL_SIZE:
        raise ValueError(f'Asking for {tot_size} MB, be kind to our poor government servers')
    
    filenames = []
    # download results
    for result in results:
        url = result[1]
        filepath = data_dir.joinpath(url.split('/')[-1])
        if not filepath.exists(): #check if already downloaded
            wget.download(url, out=str(filepath))
        filenames.append(str(filepath))

    return filenames


def process_geotiff(geotiff_input_filename, geotiff_output_filename, sinkholes_output_filename,
                    config,
                    output_geotiff=True,
                    output_geojson=True,
                    output_crs_epsg: int | None=None,
                    verbose: bool=True):
    
    geotiff_input: rasterio.io.DatasetReader = rasterio.open(geotiff_input_filename, nodata=0)
    if output_crs_epsg is None:
        if verbose:
            print(f'Input geotiff has CRS {geotiff_input.crs}. No output CRS specified, so using same CRS for output geotiff.')
    else:
        if geotiff_input.crs.to_epsg() is not output_crs_epsg:
            # convert to output_crs
            if verbose:
                print(f'Converting input geotiff from {geotiff_input.crs} to EPSG:{output_crs_epsg}.')
            new_crs = rasterio.crs.CRS.from_epsg(output_crs_epsg)
            transform, width, height = calculate_default_transform(
                geotiff_input.crs, 
                new_crs,
                geotiff_input.width,
                geotiff_input.height,
                *geotiff_input.bounds)
            kwargs = geotiff_input.meta.copy()
            kwargs.update({
                'crs': new_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # create temp file
            temp_file, temp_file_name = tempfile.mkstemp(suffix='.tif', dir=os.path.dirname(geotiff_output_filename))
            os.close(temp_file)
            with rasterio.open(temp_file_name, 'w', **kwargs) as reprojected_raster:
                reproject(
                    source=rasterio.band(geotiff_input, 1),
                    destination=rasterio.band(reprojected_raster, 1),
                    src_transform=geotiff_input.transform,
                    src_crs=geotiff_input.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=Resampling.cubic)
            
            # read temp file we just created
            geotiff_input: rasterio.io.DatasetReader = rasterio.open(temp_file_name, nodata=0)

            # delete temp file
            os.remove(temp_file_name)

            if verbose:
                print(f'Done converting input geotiff to EPSG:{output_crs_epsg}.')

    elevation = geotiff_input.read(1)
    elevation[elevation<0] = 0
    color_util = ColorUtil(config['min_depth_for_colormap'], config['max_depth_for_colormap'], config['pin_colormap'], config['map_colormap'])
    if config['verbose']:
        print('Loaded geotiff DEM.')

    # fill depressions, get difference, and get difference
    rich_dem = rd.rdarray(elevation, no_data=0)
    diff = np.array(rd.FillDepressions(rich_dem) - rich_dem)
    del rich_dem # save memory
    gc.collect()

    if config['verbose']:
        print('Done with depression filling.')

    if output_geotiff:
        # make hillshade map with sinkholes highlighted by depth
        hillshade = es.hillshade(elevation, azimuth=config['hillshade_azimuth'], altitude=config['hillshade_altitude'])
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
        del diff_colors
        gc.collect()

        output_profile = geotiff_input.profile
        output_profile.update(dtype=rasterio.uint8, count=3, nodata=0)
        with rasterio.open(geotiff_output_filename, 'w', **output_profile) as geotiff_output:
            geotiff_output.write(img)
        
        print('Exported geotiff map.')

    if output_geojson:
        if config['verbose']:
            print('Depth to pin color mapping for GaiaGPS:')
            print(color_util.gaia_colormap_string(config['units']))

        time_before_sinkholes = time.time()
        sinkholes = sinkholes_from_diff(diff, geotiff_input, elevation, config['min_depth'], config['max_dimension'])
        time_after_sinkholes = time.time()
        print(f'Found {len(sinkholes)} sinkholes. Elapsed time making sinkhole objects: {time_after_sinkholes - time_before_sinkholes:.2f} s.')
        export_sinkholes_geojson(sinkholes, sinkholes_output_filename, color_util, config)
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

def export_sinkholes_geojson(sinkholes, output_filename, color_util, config):
    if output_filename == None or output_filename == '':
        raise ValueError('Must specify an output filename')
    
    folder_uuid = uuid.uuid4()
    units = config['units']
    max_points_per_file = config['max_points_per_file']

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
            export_sinkholes_geojson(sinkholes[i * max_points_per_file : (i+1) * max_points_per_file], new_output_fname, color_util, config)
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
                                Depths range from {min_depth * unit_conversion}:.1f {unit_str} to {max_depth * unit_conversion:.1f} {unit_str}.""",
                "config": config
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Find Sinkholes', description='Automatically find sinkholes using 1m DEMs from USGS')
    parser.add_argument('-i', '--input', action='store')
    parser.add_argument('-otif', '--output-geotiff', action='store', help='Filename to output .tif file that is hillshade with sinkholes highlighted. Or a folder, in which case the filename of the input file will be used for the output.')
    parser.add_argument('-ojson', '--output-geojson', action='store', help='Filename to output .geojson file that is list of sinkholes. Or a folder, in which case the filename of the input file will be used for the output.')
    parser.add_argument('-c', '--config', action='store')
    parser.add_argument('-a', '--area', action='store')
    parser.add_argument('-q', '--qgis', action='store', help='QGIS project file to add output layers to. If this file does not exist, it will be created from ridgewalking_template.qgs in template.')
    parser.add_argument('-crs', '--crs-geotiff', action='store', type=int, help='EPSG code of coordinate reference system to use for output geotiff file. If not specified, use same CRS as input geotiff.')
    args = parser.parse_args()

    if not ('input' in args and args.input != None) and args.area is None:
        print('Must have --input (or -i) argument that specifies input .tif file with digital elevation model \
    or specify an area for which to download DEMs using --area')
        exit(1)

    # parse config file
    if 'config' in args and args.config != None:
        config = parse_config(args.config)
    else:
        config = default_config()

    if args.input is not None:
        input_file = args.input
    if args.area is not None:
        dems = download_dems(args.area)
        if dems == []:
            print('No DEMs were found for that search area')
            exit(1)
        input_file = dems[0]
    
    input_file_fname_no_ext = os.path.splitext(os.path.basename(input_file))[0]

    output_geotiff = 'output_geotiff' in args and args.output_geotiff != None
    output_geojson = 'output_geojson' in args and args.output_geojson != None

    output_geotiff_fname = ''
    output_geojson_fname = ''
    if output_geotiff:
        output_geotiff_fname = args.output_geotiff
        if os.path.isdir(output_geotiff_fname):
            possible_geotiff_fname = os.path.join(output_geotiff_fname, input_file_fname_no_ext + '.tif')
            if os.path.exists(possible_geotiff_fname) and os.path.samefile(input_file, possible_geotiff_fname):
                # avoid overwriting input file if user specifies same directory for input and output files
                output_geotiff_fname = os.path.join(output_geotiff_fname, input_file_fname_no_ext + '_output.tif')
            else:
                output_geotiff_fname = possible_geotiff_fname

    if output_geojson:
        output_geojson_fname = args.output_geojson
        if os.path.isdir(output_geojson_fname):
            output_geojson_fname = os.path.join(output_geojson_fname, input_file_fname_no_ext + '.geojson')

    if config['verbose']:
        print('Using config object:')
        print(json.dumps(config, indent=4))

    process_geotiff(input_file, output_geotiff_fname, output_geojson_fname,
                    config=config,
                    output_geotiff=output_geotiff,
                    output_geojson=output_geojson,
                    output_crs_epsg=args.crs_geotiff,
                    verbose=config['verbose'])