import argparse
import glob
import os
import json
import numpy as np
import pdal
from pprint import pprint
import requests
import subprocess
import tempfile

from settings import Settings

ERASE_LINE_CODE = '\033[2K'

# from https://www.usgs.gov/ngp-standards-and-specifications/lidar-base-specification-tables
POINT_CLOUD_CODE_TO_DESCRIPTION = {
    1: "Processed, but unclassified",
    2: "Bare earth",
    7: "Low noise",
    9: "Water",
    17: "Bridge deck",
    18: "High noise",
    20: "Ignored ground (typically breakline proximity)",
    21: "Snow (if present and identifiable)",
    22: "Temporal exclusion (typically nonfavored data in intertidal zones)"
}

class LidarProcessor:
    """Class containing methods that do stuff with LIDAR data.
    This class mainly acts as a container for settings used by all these methods,
    so that they don't have to be passed in as arguments to each method.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def download_point_clouds(self,
                              bbox_lower_left: list[float],
                              bbox_upper_right: list[float],
                              usgs_response_output_fname: str | None=None) -> list[str]:
        """
        Download point clouds for the specified bounding box and save them to the output directory.

        Parameters:
        - bbox_lower_left: A list of [latitude, longitude] defining the lower-left corner of the bounding box, in WGS84 coordinates.
        - bbox_upper_right: A list of [latitude, longitude] defining the upper-right corner of the bounding box, in WGS84 coordinates.

        Returns:
        - A list of file paths to the downloaded point clouds.
        """

        point_cloud_list_url = f'https://tnmaccess.nationalmap.gov/api/v1/products?prodFormats=LAS,LAZ&datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={bbox_lower_left[1]},{bbox_lower_left[0]},{bbox_upper_right[1]},{bbox_upper_right[0]},&'
        point_cloud_list_request = requests.request(url=point_cloud_list_url, method='GET')
        point_cloud_list = point_cloud_list_request.json()

        if usgs_response_output_fname is not None:
            with open(usgs_response_output_fname, 'w') as output_file:
                json.dump(point_cloud_list, output_file, indent=2)

        if 'items' not in point_cloud_list:
            print(f'Error fetching point cloud list for bbox {bbox_lower_left} to {bbox_upper_right}. Response is:')
            pprint(point_cloud_list)
            return []
        num_items = len(point_cloud_list['items'])
        downloaded_files = []

        for i, item in enumerate(point_cloud_list['items']):
            # get filename from end of downloadURL
            url = item['downloadURL']
            fname = url.split('/')[-1]
            if self.settings.verbose:
                print(f'\r{ERASE_LINE_CODE}Downloading point cloud {i + 1}/{num_items}: {fname}', end='')
            try:
                point_cloud_request = requests.request(url=url, method='GET')
                if point_cloud_request.status_code != 200:
                    print(f'\nError downloading {fname} from {url}: {point_cloud_request.status_code} {point_cloud_request.reason}')
                    continue
                output_fname = os.path.join(self.settings.point_clouds_dir, fname)
                with open(output_fname, 'wb') as output_file:
                    output_file.write(point_cloud_request.content)
                    downloaded_files.append(output_fname)
            except ConnectionError as e:
                print(f'\nError downloading {fname} from {url}: {e}')
            except Exception as e:
                print(f'Unexpected error downloading {fname} from {url}: {e}')
        print('')  # print newline after progress output
        return downloaded_files

    def make_dem(self, input_laz: str, output_tif: str):

        # Define the pipeline stages
        # 1. Read the file
        # 2. Filter for Classifications of interest
        # 3. Create the DEM using the minimum Z value in a grid with specified resolution
        classification_str = ', '.join([f"Classification[{c}:{c}]" for c in self.settings.point_cloud_classifications])
        pipeline_json = [
            {
                "type": "readers.las",
                "filename": input_laz
            },
            {
                "type": "filters.range",
                "limits": classification_str
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "output_type": "min",
                "resolution": self.settings.resolution,
                "nodata": self.settings.nodata_value,
                "data_type": "float32",
                "gdaldriver": "GTiff"
            }
        ]

        # Initialize and execute
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        # if self.settings.verbose:
        #     print(f'Total points processed: {pipeline.arrays[0].shape[0]}')
        #     raster_area = (pipeline.arrays[0]['X'].max() - pipeline.arrays[0]['X'].min()) * (pipeline.arrays[0]['Y'].max() - pipeline.arrays[0]['Y'].min())
        #     print(f'Points per grid cell: {pipeline.arrays[0].shape[0] / (raster_area)}')
        #     unique_elements, counts_elements = np.unique(pipeline.arrays[0]['Classification'], return_counts=True)
        #     for val, count in zip(unique_elements, counts_elements):
        #         print(f'Class {val:d} {POINT_CLOUD_CODE_TO_DESCRIPTION[val]}')
        #         print(f'    Count: {count}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do stuff with point clouds and DEMs.')
    parser.add_argument('-ll', '--bbox-lower-left', type=str, help='Lower-left corner of the bounding box, in the form latitude,longitude', required=False)
    parser.add_argument('-ur', '--bbox-upper-right', type=str, help='Upper-right corner of the bounding box, in the form latitude,longitude', required=False)
    parser.add_argument('-pc', '--point-clouds', type=str, nargs='*', help='Path to a point cloud file to process into a DEM. Specify multiple point cloud files with globstars.', required=False)
    parser.add_argument('-s', '--settings', type=str, help='Path to a JSON configuration file with settings for the DEM generation process', required=False)

    args = parser.parse_args()

    settings = Settings.from_json(args.settings) if args.settings else Settings()
    lidar_processor = LidarProcessor(settings)

    point_cloud_files = []

    # Download point clouds using lower left and upper right coordinates if provided
    bbox_lower_left = [float(coord) for coord in args.bbox_lower_left.split(',')] if args.bbox_lower_left else None
    bbox_upper_right = [float(coord) for coord in args.bbox_upper_right.split(',')] if args.bbox_upper_right else None

    if args.bbox_lower_left is not None and args.bbox_upper_right is not None:
        point_cloud_files = lidar_processor.download_point_clouds(bbox_lower_left,
                                                                       bbox_upper_right,
                                                                       usgs_response_output_fname='response.json')
        print(f'Downloaded {len(point_cloud_files)} point cloud files.')

    # Find point cloud files using glob if provided
    if args.point_clouds is not None:
        for file in args.point_clouds:
            point_cloud_files.append(file)

    num_point_clouds = len(point_cloud_files)

    # Process point clouds into DEMs
    dem_fnames = []
    for point_cloud_index, point_cloud_fname in enumerate(point_cloud_files):
        dem_fname = os.path.join(settings.dems_dir, f'{point_cloud_fname.split('/')[-1].split('.')[0]}.tif')
        print(f'\r{ERASE_LINE_CODE}Processing point cloud {point_cloud_index + 1}/{num_point_clouds} into DEM at {dem_fname}', end='', flush=True)
        lidar_processor.make_dem(input_laz=point_cloud_fname,
                                    output_tif=dem_fname)
        dem_fnames.append(dem_fname)
        
    print(f'\nDone processing {num_point_clouds} point clouds into DEMs.')
    
    exit(0)
    for dem_fname in dem_fnames:
        subprocess_args = ['process_dem', dem_fname]
        if args.settings is not None:
            subprocess_args.append('--settings')
            subprocess_args.append(args.settings)
        subprocess.call(subprocess_args)