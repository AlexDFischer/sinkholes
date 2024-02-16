import os
import numpy as np
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import richdem as rd
from matplotlib.colors import LinearSegmentedColormap
from pyrsgis import raster

datasource, elevation = raster.read("lonesomeRidgeArea.tif")

elevation[elevation<0] = 0

hillshade = es.hillshade(elevation, azimuth=315, altitude=30)

rich = rd.rdarray(elevation, no_data=0)
richFilled = rd.FillDepressions(rich)

# make my own colormap with 0 being transparent
ncolors = 256
color_array = plt.get_cmap('inferno_r')(range(ncolors))

# change alpha values
color_array[0,3] = 0 # make this colormap be transparent for value 0

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='inferno_r_alpha',colors=color_array)

# register this new colormap with matplotlib
plt.colormaps.register(cmap=map_object)


diff = richFilled - rich

fig, ax = plt.subplots(figsize=(15, 9))
ax.imshow(hillshade, cmap="Greys_r")
ep.plot_bands(diff, ax=ax, alpha=0.5, cmap="inferno_r_alpha", title="test")
plt.ylabel("depth (meters)")
plt.show()

# diffs_nonzero = diff[diff != 0]

# plt.hist(diffs_nonzero.flatten(), bins=30)
# plt.yscale("log")
# plt.title("Depth of sinkholes found by sinkhole-filling difference algorithm.\n10km by 10km square in southern Guadalupe Mountains")
# plt.xlabel("depth of sinkhole (m)")
# plt.ylabel("Number of 1m squares with that depth")
# plt.show()

