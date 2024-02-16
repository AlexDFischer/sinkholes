#!/bin/python3
import units
import units.predefined
from units import unit
from datetime import datetime
from math import log10
import matplotlib.pyplot as plt
import numpy as np

units.predefined.define_units()
meter = units.predefined.unit('m')(1)

def rgb_to_hex(rgb):
    """input: numpy ndarray with 3 values, red, green, and blue, which are floats in [0, 1]
    Return color as #RRGGBB for the given color values."""
    rgb = (256*rgb).astype(np.int)
    return '#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2])

ncolors = 256
color_array = [rgb_to_hex(rgb) for rgb in plt.get_cmap('gist_rainbow')(range(ncolors))]

# depth must be  quantity with UNITS
# color uses gist_rainbow colormap from matplotlib, with depth on a log scale frmo 1m to 10m
# smaller or larger values are clipped to 1 or 10m
def depth_to_hex_color(depth):
    depth_m = depth / meter

    # log scale from 1 to 10 m
    log_num = log10(max(depth_m, 1))
    log_num = min(log_num, 1)
    # now log_num goes from 0 to 1

    index = int(log_num * (ncolors - 1))
    # make sure floating point weirdness doesn't screw us
    index = min(index, ncolors-1)
    index = max(index, 0)

    return color_array[index]


class Sinkhole:

    def __init__(self, depth, lat, long, width=0, length=0, elevation=None, area=0, units='metric'):
        if units == 'metric':
            length_unit = unit('m')
        elif units == 'imperial':
            length_unit = unit('ft')
        else:
            raise ValueError(f'Error: `"units`" was \"{units}\", but the only allowed values are \"metric\" or \"imperial\".')
        
        self.depth = length_unit(depth)
        self.lat = lat
        self.long = long
        self.width = length_unit(width)
        self.length = length_unit(length)
        self.elevation = length_unit(elevation)
        self.area = area * length_unit * length_unit
        self.time = datetime.now()
    
    def depth_to_hex_color(self):
        pass
    
    def json_obj(self, units='metric'):
        length_unit = None
        if units == 'metric':
            length_unit = unit('m')(1)
        elif units == 'imperial':
            length_unit = unit('ft')(1)
        else:
            raise ValueError(f'Error: `"units`" was \"{units}\", but the only allowed values are \"metric\" or \"imperial\".')
        
        title = "sinkhole {:.1f} d".format(self.depth / length_unit)
        if self.width != 0 and self.length != 0:
            title += "{:.1f}w {:.1f}l".format(self.width / length_unit, self.length/length_unit)
        
        time_str = self.time.strftime("%Y-%M-%dT%H:%M:%SZ")
        
        {
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
                "notes": f"elevation: {str(self.elevation)}\narea: {str(self.area)}",
                "latitude": self.lat,
                "longitude": self.long,
                "elevation": int(self.elevation / length_unit),
                "marker_type": "pin",
                "marker_color": depth_to_hex_color(self.depth),
                "marker_decoration": None
            }
        }