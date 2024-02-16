import numpy as np
import matplotlib.pyplot as plt
import earthpy.spatial as es
import richdem as rd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyrsgis import raster
import cv2
from sinkhole import Sinkhole
import json
import time
from osgeo import osr, gdal

def exportGeoTif(input_filename, output_filename, colormap='inferno_r', alpha=0.7, show=False):
    datasource, elevation = raster.read(input_filename)
    elevation[elevation<0] = 0
    hillshade = es.hillshade(elevation, azimuth=315, altitude=30)

    # make colormap without alpha channel, and scale it to be unsigned byte
    ncolors = 256
    color_array = plt.get_cmap(colormap)(range(ncolors))
    color_array = color_array[:, :3] # remove the alpha channel
    color_array = (color_array * 255).astype(np.uint8)

    print("Loaded geotiff and made colormap")

    # fill depressions, get difference, and get difference that is unsigned byte scaled 0-255
    rich_dem = rd.rdarray(elevation, no_data=0)
    diff = rd.FillDepressions(rich_dem) - rich_dem
    rich_dem = None
    max_diff = np.max(diff)
    scaled_diff = (diff * 255 / max_diff).astype(np.uint8)

    print("Done with depression filling")

    # image to export and render
    img = np.zeros(shape=(hillshade.shape[0], hillshade.shape[1], 3), dtype=np.ubyte)
    for channel in range(3):
        img[:, :, channel] = hillshade

    scaled_diff_colored = np.zeros_like(img)
    scaled_diff_colored[:, :, :] = color_array[scaled_diff, :]

    nonzero_diff_index = diff > 0

    img[nonzero_diff_index, :] = scaled_diff_colored[nonzero_diff_index, :]

    # save some memory
    scaled_diff = None
    scaled_diff_colored = None
    nonzero_diff_index = None

    print("made composite image")

    time_before_sinkholes = time.time()
    sinkholes = sinkholes_from_diff(diff, datasource, elevation)
    for sinkhole in sinkholes:
        print(json.dumps(sinkhole.json_obj(), indent=4))
    time_after_sinkholes = time.time()
    print(f"found total of {len(sinkholes)} sinkholes")
    print(f"Made sinkhole objects. Elapsed time making sinkhole objects: {time_after_sinkholes - time_before_sinkholes} s")
                                                   
    # export geotiff TODO this runs out of memory, split up into multiple files
    # raster.export(img, datasource, output_filename)

    if show:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
        plt.colorbar(ScalarMappable(norm=Normalize(0, max_diff), cmap=colormap))
        plt.show()
    

def sinkholes_from_diff(diff, datasource, elevation, min_depth=0.7, max_dimension=300):
    """max dimension is the maximum width or length allowed for a sinkhole before we no longer include it"""
    diffs_nonzero = ((diff >= min_depth) * 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diffs_nonzero, connectivity=4, ltype=cv2.CV_16U)

    sinkholes = []

    max_depths = np.zeros((num_labels))
    for x in range(diff.shape[0]):
        for y in range(diff.shape[1]):
            depth = diff[x,y]
            label = labels[x,y]
            if depth > max_depths[label]:
                max_depths[label] = depth
    
    for label in range(num_labels):
        width = stats[label, cv2.CC_STAT_WIDTH]
        length = stats[label, cv2.CC_STAT_HEIGHT]
        if width <= max_dimension and length <= max_dimension:
            x = centroids[label, 0]
            y = centroids[label, 1]
            wgs84_point = pixel_to_wgs84_coords(x, y, datasource)
            x = int(round(x))
            y = int(round(y))
            sinkhole = Sinkhole(depth=max_depths[label],
                                lat=wgs84_point[0],
                                long=wgs84_point[1],
                                width=width,
                                length=length,
                                elevation=elevation[x,y]+diff[x,y],
                                area=cv2.CC_STAT_AREA)
            sinkholes.append(sinkhole)

    return sinkholes

def pixel_to_wgs84_coords(x, y, datasource):
    """x, y is the pixel location in the geotiff we want to convert to WGS84 coordinates.
    datasource is the gdal datasource obtained from loading the geotiff using gdal."""
    
    geo_transform = datasource.GetGeoTransform()

    # Convert pixel coordinates to georeferenced coordinates
    x_geo = geo_transform[0] + x*geo_transform[1] + y*geo_transform[2]
    y_geo = geo_transform[3] + x*geo_transform[4] + y*geo_transform[5]

    # Build the coordinate transformation
    old_coords = osr.SpatialReference()
    old_coords.ImportFromWkt(datasource.GetProjection())
    new_coords = osr.SpatialReference()
    new_coords.ImportFromEPSG(4326)  # EPSG code for WGS84
    transform = osr.CoordinateTransformation(old_coords, new_coords)

    # Transform georeferenced coordinates to GPS coordinates
    return transform.TransformPoint(x_geo, y_geo)

def export_sinkholes_geojson(sinkholes, output_filename):
    pass


exportGeoTif("lonesomeRidgeArea.tif", "output/lonesomeRidgeArea.tif", show=False)
