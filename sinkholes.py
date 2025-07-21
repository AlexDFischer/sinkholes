#!/bin/python3

from datetime import datetime
import math
import numpy as np
import earthpy.spatial as es
import gc
import json
import pyproj
import rasterio
import richdem as rd
import cv2
import pprint
import pyjson5
import time
import argparse
import traceback
import uuid
from pathlib import Path
import requests
import wget

from config import Config
from sinkhole import Sinkhole
from util import feet_per_meter, gaia_datetime_format
from color_utils import ColorUtil

wgs84_name = 'EPSG:4326'


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
    # can also set to [opr_dataset, dem_1m_dataset] if higher res desired
    results = []
    for dataset in [dem_1m_dataset]:
        url = f"https://tnmaccess.nationalmap.gov/api/v1/products?&datasets={dataset}&bbox={bounding_box}&prodFormats=&max=1000&offset=0"

        try:
            result = requests.get(url=url, timeout=10)
        except requests.exceptions.ReadTimeout:
            print(f"{url} failed to elicit a reply")
            raise
        results = [(item['title'], item['downloadURL'])
                   for item in json.loads(result.text)["items"]]
        if len(results) > 0:
            break

    if results:
        # TODO: for now, just download the first result. In the long run, join all results and crop to bounding box
        results = [results[0]]

    # check total download size to make sure we aren't accidentally overloading the servers
    tot_size = 0
    MAX_DL_SIZE = 500  # in megabytes
    for result in results:
        response = requests.head(result[1], allow_redirects=True)
        size = int(response.headers.get('content-length', -1), )
        tot_size += size
    tot_size = tot_size / float(1 << 20)  # convert to MB
    if tot_size > MAX_DL_SIZE:
        raise ValueError(
            f'Asking for {tot_size} MB, be kind to our poor government servers')

    filenames = []
    # download results
    for result in results:
        url = result[1]
        filepath = data_dir.joinpath(url.split('/')[-1])
        if not filepath.exists():  # check if already downloaded
            wget.download(url, out=str(filepath))
        filenames.append(str(filepath))

    return filenames


def process_geotiff(config: Config,
                    geotiff_input_filename: str | None = None,
                    geotiff_output_filename: str | None = None,
                    sinkholes_output_filename: str | None = None,
                    database_output_filename: str | None = None):

    geotiff_input = rasterio.open(geotiff_input_filename, nodata=0)
    elevation = geotiff_input.read(1)
    elevation[elevation < 0] = 0
    color_util = ColorUtil(config.min_depth_for_colormap,
                           config.max_depth_for_colormap,
                           config.pin_colormap,
                           config.map_colormap)
    if config.verbose:
        print('Loaded geotiff DEM.')

    # fill depressions, and get difference
    rich_dem = rd.rdarray(elevation, no_data=0)
    diff = np.array(rd.FillDepressions(rich_dem) - rich_dem)
    del rich_dem  # save memory
    gc.collect()

    if config.verbose:
        print('Done with depression filling.')

    if geotiff_output_filename != None:
        # make hillshade map with sinkholes highlighted by depth
        hillshade = es.hillshade(
            elevation, azimuth=config.hillshade_azimuth, altitude=config.hillshade_altitude)
        img = np.zeros(
            shape=(3, hillshade.shape[0], hillshade.shape[1]), dtype=np.uint8)
        for channel in range(3):
            # img has all 3 channels equal to hillshade map (so img is black and white hillshade map)
            img[channel, :, :] = hillshade

        del hillshade  # save some memory
        gc.collect()

        map_color_ufunc = np.frompyfunc(color_util.depth_to_map_color, 1, 1)
        nonzero_diff_index = diff > 0
        # need to stack because numpy wants to store the output of map_color_ufunc as an array of arrays
        diff_colors = np.stack(map_color_ufunc(diff[nonzero_diff_index]))
        # rearrange axes so last index is index for 3 color channels
        diff_colors = np.moveaxis(diff_colors, -1, 0)
        img[:, nonzero_diff_index] = diff_colors

        del nonzero_diff_index  # save some memory
        del diff_colors
        gc.collect()

        output_profile = geotiff_input.profile
        output_profile.update(dtype=rasterio.uint8, count=3, nodata=0)
        with rasterio.open(geotiff_output_filename, 'w', **output_profile) as geotiff_output:
            geotiff_output.write(img)

        print('Exported geotiff map.')

    if sinkholes_output_filename != None or database_output_filename != None:
        if config.verbose:
            print('Depth to pin color mapping for GaiaGPS:')
            print(color_util.gaia_colormap_string(config.units))

        time_before_sinkholes = time.time()
        sinkholes = sinkholes_from_diff(diff, geotiff_input, elevation, config.min_depth, config.max_dimension)
        time_after_sinkholes = time.time()
        print(f'Found {len(sinkholes)} sinkholes. Elapsed time making sinkhole objects: {time_after_sinkholes - time_before_sinkholes:.2f} s.')
        if sinkholes_output_filename != None:
            export_sinkholes_geojson(
                sinkholes, sinkholes_output_filename, color_util, config)
            print('Exported sinkhole objects to geojson file(s).')
        if database_output_filename != None:
            # TODO
            pass


def sinkholes_from_diff(diff,
                        geotiff_input,
                        elevation,
                        min_depth: float,
                        max_dimension: float) -> list[Sinkhole]:
    """max dimension is the maximum width or length allowed for a sinkhole before we no longer include it"""

    print('Entered sinkholes_from_diff')

    # make wgs84_transformer object, that transforms from the built in coordinate reference system of the geotiff we're reading
    wgs84_transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(str(geotiff_input.crs)), pyproj.CRS(wgs84_name))
    print('Constructed wgs84_transformer')

    diffs_nonzero = (diff > 0).astype(np.uint8)
    gc.collect() # Above expression generates lots of big numpy arrays we don't need

    print('Computed nonzero diffs')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
        diffs_nonzero,
        connectivity=4,
        ltype=cv2.CV_16U,
        ccltype=cv2.CCL_WU)

    print(f'Found {num_labels} connected components in diff')
    sinkholes: list[Sinkhole] = []

    max_depths = np.zeros((num_labels))
    max_depth_locations = np.zeros((num_labels, 2))
    for row in range(diff.shape[0]):
        for col in range(diff.shape[1]):
            depth = diff[row, col]
            label = labels[row, col]
            if depth > max_depths[label]:
                max_depths[label] = depth
                max_depth_locations[label] = [row, col]
    print('Computed max depths and locations for each label')

    for label in range(num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        length = stats[label, cv2.CC_STAT_HEIGHT]
        if max_depths[label] >= min_depth and width <= max_dimension and length <= max_dimension:
            row = max_depth_locations[label, 0]
            col = max_depth_locations[label, 1]
            input_crs_coords = geotiff_input.transform * (col, row)
            lat, long = wgs84_transformer.transform(
                input_crs_coords[0], input_crs_coords[1])
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

    print('Returned sinkholes from diff')
    return sinkholes


def export_sinkholes_geojson(sinkholes: list[Sinkhole], output_filename: str, color_util: ColorUtil, config: Config):
    if output_filename == '':
        raise ValueError('Must specify an output filename')

    folder_uuid = uuid.uuid4()
    units = config.units
    max_points_per_file = int(config.max_points_per_file)

    if max_points_per_file > 0 and len(sinkholes) > max_points_per_file:
        # split up the sinkholes into multiple files
        fname_arr = output_filename.split('.')
        fname_prefix = ''  # part of filename before extension
        fname_extension = ''
        if len(fname_arr) == 1:
            # there's no file extension
            fname_prefix = output_filename
        else:
            fname_prefix = '.'.join(fname_arr[:-1])
            fname_extension = '.' + fname_arr[-1]

        num_files = (len(sinkholes) + max_points_per_file -
                     1) // max_points_per_file
        num_decimal_digits = math.ceil(math.log10(num_files))
        for i in range(num_files):
            new_output_fname = fname_prefix + \
                ('_{:0' + str(num_decimal_digits) + 'n}').format(i) + fname_extension
            export_sinkholes_geojson(sinkholes[i * max_points_per_file: (i+1) *
                                     max_points_per_file], new_output_fname, color_util, config)
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
            raise ValueError(
                f'Error: "units" was "{units}", but the only allowed values are "metric" or "imperial".')

        now = datetime.now()
        min_depth: float = min([sinkhole.depth for sinkhole in sinkholes])
        max_depth: float = max([sinkhole.depth for sinkhole in sinkholes])
        output: dict[str, object] = {
            "type": "FeatureCollection",  # folder info for GaiaGPS
            "properties": {
                "name": "caves.science automatic sinkholes",
                "updated_date": now.strftime(gaia_datetime_format),
                "time_created": now.strftime(gaia_datetime_format),
                "notes": f"""{len(sinkholes)} sinkholes automatically detected by caves.science.\n                                Depths range from {min_depth * unit_conversion}:.1f {unit_str} to {max_depth * unit_conversion:.1f} {unit_str}.""",
                "config": vars(config)
            },
            "features": [{  # folder info for Caltopo
                "geometry": None,
                "id": folder_uuid.hex,
                "type": "Feature",
                "properties": {
                    "visible": True,
                    "title": "caves.science automatic sinkholes",
                    "class": "Folder",
                    "labelVisible": True
                }
            }] + [sinkhole.json_obj(color_util, folder_uuid, units=units) for sinkhole in sinkholes]
        }

        file = open(output_filename, 'w')
        file.write(json.dumps(output, indent=4))


parser = argparse.ArgumentParser(
    prog="Find Sinkholes", description="Automatically find sinkholes using 1m DEMs from USGS")
parser.add_argument('-i', '--input', default=None)
parser.add_argument('-otif', '--output-geotiff', default=None)
parser.add_argument('-ojson', '--output-geojson', default=None)
parser.add_argument('-odb', '--output-database', default=None)
parser.add_argument('-c', '--config', default=None)
parser.add_argument('--area', default=None)
args = parser.parse_args()

if args.input is None and args.area is None:
    print('Must have --input (or -i) argument that specifies input .tif file with digital elevation model, or specify an area for which to download DEMs using --area.')
    exit(1)

if args.output_geotiff is None and args.output_geojson is None and args.output_database is None:
    print('Error: must specify an output via --output-geotiff, --output-geojson, or --output-database.')
    exit(1)

# parse config file
if args.config is None:
    config: Config = Config.default_config()
else:
    config: Config = Config.from_file(args.config)

input_file: str = ''
if args.input is not None:
    input_file = args.input
elif args.area is not None:
    dems = download_dems(args.area)
    if dems == []:
        print('No DEMs were found for that search area')
        exit(1)
    input_file = dems[0]
else:
    print(f'Error: no input file specified, and no area specified to download DEMs for. Either one must be specified via --input or --area.')
    exit(1)

output_geotiff = 'output_geotiff' in args and args.output_geotiff != None
output_geojson = 'output_geojson' in args and args.output_geojson != None

output_geotiff_fname = ''
output_geojson_fname = ''
if output_geotiff:
    output_geotiff_fname = args.output_geotiff
if output_geojson:
    output_geojson_fname = args.output_geojson

if config.verbose:
    print('Using config object:')
    pprint.pprint(vars(config))

process_geotiff(config=config,
                geotiff_input_filename=input_file,
                geotiff_output_filename=args.output_geotiff,
                sinkholes_output_filename=args.output_geojson,
                database_output_filename=args.output_database)
