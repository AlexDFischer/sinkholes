#!/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt
from util import feet_per_meter

pin_filenames = ['red-pin.png','orange-pin.png','yellow-pin.png','green-pin.png','blue-pin-down.png','purple-pin.png']
num_pin_filenames = len(pin_filenames)
num_colors = 256

class ColorUtil:
    def __init__(self, min_depth_for_colormap, max_depth_for_colormap, pin_colormap, map_colormap):
        self.min_depth_for_colormap = min_depth_for_colormap
        self.max_depth_for_colormap = max_depth_for_colormap
        self.log_depth_diff = math.log(max_depth_for_colormap) - math.log(min_depth_for_colormap)
        self.depth_diff = max_depth_for_colormap - min_depth_for_colormap

        self.color_array_pin = plt.get_cmap(pin_colormap)(range(num_colors))
        self.color_array_pin = self.color_array_pin[:, :3] # remove the alpha channel
        self.color_array_pin = np.rint(self.color_array_pin * 255).astype(np.uint8) # convert to uint8

        self.color_array_map = plt.get_cmap(map_colormap)(range(num_colors))
        self.color_array_map = self.color_array_map[:, :3] # remove the alpha channel
        self.color_array_map = np.rint(self.color_array_map * 255).astype(np.uint8) # convert to uint8
    
    def depth_to_pin_color(self, depth):
        """Convert the depth to a color, using a log scale that goes between min_depth_for_colormap and max_depth_for_colormap.
        Returns a color, in the form of a 3-tuple of uint8's in [0, 255].
        """
        if depth <= self.min_depth_for_colormap:
            return self.color_array_pin[0]
        
        if depth >= self.max_depth_for_colormap:
            return self.color_array_pin[-1]
        
        index = round(np.log(depth / self.min_depth_for_colormap) / self.log_depth_diff * (num_colors - 1))
        index = max(index, 0)
        index = min(index, num_colors - 1)
        return self.color_array_pin[index]
    
    def depth_to_pin_filename(self, depth):
        """
        Returns one of the following filenames, which are those available in GaiaGPS:
            red-pin.png
            orange-pin.png
            yellow-pin.png
            green-pin.png
            blue-pin-down.png
            purple-pin.png
        using a log scale that goes between min_depth_for_colormap and max_depth_for_colormap.
        Depth can be a numpy array.
        """
        if depth <= self.min_depth_for_colormap:
            return pin_filenames[0]
        
        if depth >= self.max_depth_for_colormap:
            return pin_filenames[-1]
        
        index = round(np.log(depth / self.min_depth_for_colormap) / self.log_depth_diff * (num_pin_filenames - 1))
        index = max(index, 0)
        index = min(index, num_pin_filenames - 1)
        return pin_filenames[index]
    
    def depth_to_map_color(self, depth):
        """
        Convert the depth to a color, using a linear scale that ends at max_depth_for_colormap.

        Returns a color, in the form of a length 3 numpy array of uint8's in [0, 255].
        """
        if depth >= self.max_depth_for_colormap:
            return self.color_array_map[-1]
        
        index = round((depth - self.min_depth_for_colormap) / self.depth_diff * (num_colors - 1))
        index = max(index, 0)
        index = min(index, num_colors - 1)
        return self.color_array_map[index]
    
    def gaia_colormap_string(self, units='metric'):
        """
        Returns a string that describes the map between depth and GaiaGPS pin colors.
        """
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
        
        num_pin_filenames = len(pin_filenames)
        max_min_ratio = self.max_depth_for_colormap / self.min_depth_for_colormap
        result = ''
        for i, pin_filename in enumerate(pin_filenames):
            if i == 0:
                result += f'Using pin filename {pin_filename} for 0 <= depth < {self.min_depth_for_colormap * max_min_ratio**((i+0.5)/num_pin_filenames) * unit_conversion:.2f} {unit_str}\n'
            elif i == num_pin_filenames - 1:
                result += f'Using pin filename {pin_filename} for {self.min_depth_for_colormap * max_min_ratio**((i-0.5)/num_pin_filenames) * unit_conversion:.2f} {unit_str} <= depth'
            else:
                result += f'Using pin filename {pin_filename} for {self.min_depth_for_colormap * max_min_ratio**((i-0.5)/num_pin_filenames) * unit_conversion:.2f} {unit_str} <= depth < {self.min_depth_for_colormap * max_min_ratio**((i+0.5)/num_pin_filenames) * unit_conversion:.2f} {unit_str}\n'
        return result