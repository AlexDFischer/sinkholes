import os
import numpy as np
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import richdem as rd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyrsgis import raster
from units import unit
from sinkhole import Sinkhole

def exportGeoTif(input_filename, output_filename, colormap='inferno_r', alpha=0.7, show=False):
    datasource, elevation = raster.read(input_filename)
    elevation[elevation<0] = 0
    hillshade = es.hillshade(elevation, azimuth=315, altitude=30)

    # make colormap without alpha channel, and scale it to be unsigned byte
    ncolors = 256
    color_array = plt.get_cmap(colormap)(range(ncolors))
    color_array = color_array[:, :3] # remove the alpha channel
    color_array = (color_array * 255).astype(np.uint8)

    # fill depressions, get difference, and get difference that is unsigned byte scaled 0-255
    rich_dem = rd.rdarray(elevation, no_data=0)
    diff = rd.FillDepressions(rich_dem) - rich_dem
    max_diff = np.max(diff)
    scaled_diff = (diff * 255 / max_diff).astype(np.uint8)

    # image to export and render
    img = np.zeros(shape=(hillshade.shape[0], hillshade.shape[1], 3), dtype=np.ubyte)
    for channel in range(3):
        img[:, :, channel] = hillshade

    scaled_diff_colored = np.zeros_like(img)
    scaled_diff_colored[:, :, :] = color_array[scaled_diff, :]

    nonzero_diff_index = diff > 0

    img[nonzero_diff_index, :] = scaled_diff_colored[nonzero_diff_index, :]
    # img[nonzero_diff_index] += alpha * scaled_diff_colored[nonzero_diff_index]

    # export geotiff TODO this runs out of memory, split up into multiple files
    # raster.export(img, datasource, output_filename)

    if show:
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.imshow(img)
        plt.colorbar(ScalarMappable(norm=Normalize(0, max_diff), cmap=colormap))
        plt.show()

def sinkholes_from_diff(diff):
    pass





exportGeoTif("lonesomeRidgeArea.tif", "output/lonesomeRidgeArea.tif", show=True)
