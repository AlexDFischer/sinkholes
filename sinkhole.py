#!/bin/python3

from datetime import datetime
from math import log10
import matplotlib.pyplot as plt
import numpy as np
from util import feet_per_meter, gaia_datetime_format, meters_per_foot

def rgb_entry_to_hex(rgb):
    """input: float in [0, 1]
    Return integer in [0, 255]."""
    result = int(round(rgb * 255))
    result = min(result, 255)
    result = max(result, 0)
    return result

def rgb_to_hex(rgb):
    """input: array with 3 values, red, green, and blue, which are floats in [0, 1]
    Return color as #RRGGBB for the given color values."""
    return '#%02X%02X%02X' % (rgb_entry_to_hex(rgb[0]), rgb_entry_to_hex(rgb[1]), rgb_entry_to_hex(rgb[2]))

def depth_to_pin_png(depth):
    if depth <= 0.7:
        return 'red-pin.png'
    elif depth <= 1.5:
        return 'orange-pin.png'
    elif depth <= 2.5:
        return 'yellow-pin.png'
    elif depth <= 4:
        return 'green-pin.png'
    elif depth <= 6:
        return 'blue-pin-down.png'
    else:
        return 'purple-pin.png'

ncolors = 256
color_array = [rgb_to_hex(rgb) for rgb in plt.get_cmap('gist_rainbow')(range(ncolors))]

# color uses gist_rainbow colormap from matplotlib, with depth on a log scale frmo 1m to 10m
# smaller or larger values are clipped to 1 or 10m
def depth_to_hex_color(depth):
    # log scale from 1 to 10 m
    log_num = log10(max(depth, 1))
    log_num = min(log_num, 1)
    # now log_num goes from 0 to 1

    index = int(round(log_num * (ncolors - 1)))
    # make sure floating point weirdness doesn't screw us
    index = min(index, ncolors-1)
    index = max(index, 0)

    return color_array[index]


class Sinkhole:

    def __init__(self, depth, lat, long, width=0, length=0, elevation=None, area=0, units='metric'):
        unit_conversion = None
        if units == 'metric':
            unit_conversion = 1.0
        elif units == 'imperial':
            unit_conversion = meters_per_foot
        else:
            raise ValueError(f'Error: `"units`" was \"{units}\", but the only allowed values are \"metric\" or \"imperial\".')
        
        self.depth = depth * unit_conversion
        self.lat = lat
        self.long = long
        self.width = width * unit_conversion
        self.length = length * unit_conversion
        self.elevation = elevation * unit_conversion
        self.area = area * unit_conversion * unit_conversion
        self.time = datetime.now()
    
    def depth_to_hex_color(self):
        pass
    
    def json_obj(self, units='metric'):
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
        
        title = "sinkhole {:.1f}d".format(self.depth * unit_conversion)
        if self.width != 0 and self.length != 0:
            title += " {:.1f}w {:.1f}l".format(self.width * unit_conversion, self.length * unit_conversion)
        
        time_str = self.time.strftime(gaia_datetime_format)
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    self.long,
                    self.lat
                ]
            },
            "properties": {
                "updated_date": time_str,
                "time_created": time_str,
                "deleted": False,
                "title": title,
                "is_active": True,
                "icon": depth_to_pin_png(self.depth),
                "notes": f"elevation: {int(round(self.elevation * unit_conversion))} {unit_str}\narea: {self.area*unit_conversion**2} {unit_str}^2",
                "latitude": self.lat,
                "longitude": self.long,
                "elevation": self.elevation,
                "marker_type": "pin",
                "marker_color": depth_to_hex_color(self.depth),
                "marker_decoration": None
            }
        }