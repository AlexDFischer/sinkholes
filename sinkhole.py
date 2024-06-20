#!/bin/python3

from datetime import datetime
from util import feet_per_meter, gaia_datetime_format, meters_per_foot

def rgb_to_hex(rgb):
    """input: array with 3 values, red, green, and blue, which are floats in [0, 1]
    Return color as #RRGGBB for the given color values."""
    return '#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2])


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
    
    def json_obj(self, color_util, folder_uuid, units='metric'):
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
                "icon": color_util.depth_to_pin_filename(self.depth),
                "notes": f"elevation: {int(round(self.elevation * unit_conversion))} {unit_str}\narea: {self.area*unit_conversion**2} {unit_str}^2",
                "latitude": self.lat,
                "longitude": self.long,
                "elevation": self.elevation,
                "marker_type": "pin",
                "marker-color": rgb_to_hex(color_util.depth_to_pin_color(self.depth)),
                "folderId": folder_uuid.hex
            }
        }