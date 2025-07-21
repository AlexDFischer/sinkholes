import traceback
import pyjson5

from util import meters_per_foot


class Config:
    def __init__(self,
                 units: str = "metric",
                 min_depth: float = 0.5,
                 max_dimension: float = 300,
                 min_depth_for_colormap: float = 0.5,
                 max_depth_for_colormap: float = 6,
                 max_points_per_file: int = -1,
                 pin_colormap: str = "gist_rainbow",
                 map_colormap: str = "inferno_r",
                 hillshade_azimuth: int = 315,
                 hillshade_altitude: int = 30,
                 verbose: bool = True):
        self.units = units
        self.min_depth = min_depth
        self.max_dimension = max_dimension
        self.min_depth_for_colormap = min_depth_for_colormap
        self.max_depth_for_colormap = max_depth_for_colormap
        self.max_points_per_file = max_points_per_file
        self.pin_colormap = pin_colormap
        self.map_colormap = map_colormap
        self.hillshade_azimuth = hillshade_azimuth
        self.hillshade_altitude = hillshade_altitude
        self.verbose = verbose

    @staticmethod
    def default_config():
        return Config()
    
    @staticmethod
    def from_file(fname: str) -> "Config":
        config = Config()
        unit_conversion_constant = 1.0
        try:
            with open(fname, 'r') as config_file:
                config_json = pyjson5.loads(config_file.read())
                if 'units' in config_json:
                    units = config_json['units']
                    if units == 'metric':
                        config.units = units
                    elif units == 'imperial':
                        config.units = units
                        unit_conversion_constant = meters_per_foot
                    else:
                        print(
                            f'Option units in config file "{units}" is invalid: must be "metric" or "imperial". Defaulting to {config.units}.')
                if 'min_depth' in config_json:
                    min_depth = config_json['min_depth']
                    if type(min_depth) in (int, float) and min_depth >= 0:
                        config.min_depth = min_depth * unit_conversion_constant
                    else:
                        print(
                            f'Option min_depth in config file "{str(min_depth)}" is invalid: must be nonnegative number. Defaulting to {config.min_depth}.')
                if 'max_dimension' in config_json:
                    max_dimension = config_json['max_dimension']
                    if type(max_dimension) in (int, float) and max_dimension > 0:
                        config.max_dimension = max_dimension * unit_conversion_constant
                    else:
                        print(
                            f'Option max_dimension in config file "{str(max_dimension)}" is invalid: must be positive number. Defaulting to {config.max_dimension}.')
                if 'min_depth_for_colormap' in config_json:
                    min_depth_for_colormap = config_json['min_depth_for_colormap']
                    if type(min_depth_for_colormap) in (int, float) and min_depth_for_colormap > 0:
                        config.min_depth_for_colormap = min_depth_for_colormap * unit_conversion_constant
                    else:
                        print(
                            f'Option min_depth_for_colormap in config file "{str(min_depth_for_colormap)}" is invalid: must be positive number. Defaulting to {config.min_depth_for_colormap}.')
                if 'max_depth_for_colormap' in config_json:
                    max_depth_for_colormap = config_json['max_depth_for_colormap']
                    if type(max_depth_for_colormap) in (int, float) and max_depth_for_colormap > 0:
                        config.max_depth_for_colormap = max_depth_for_colormap * unit_conversion_constant
                    else:
                        print(
                            f'Option max_depth_for_colormap in config file "{str(max_depth_for_colormap)}" is invalid: must be positive number. Defaulting to {config.max_depth_for_colormap}.')
                if 'max_points_per_file' in config_json:
                    max_points_per_file = config_json['max_points_per_file']
                    if type(max_points_per_file) == int:
                        config.max_points_per_file = max_points_per_file
                    else:
                        print(
                            f'Option max_points_per_file in config file "{str(max_points_per_file)}" is invalid: must be an integer. Defaulting to {config.max_points_per_file}.')
                if 'pin_colormap' in config_json and type(config_json['pin_colormap']) is str:
                    config.pin_colormap = config_json['pin_colormap']
                if 'map_colormap' in config_json and type(config_json['map_colormap']) is str:
                    config.map_colormap = config_json['map_colormap']
                if 'hillshade_azimuth' in config_json:
                    hillshade_azimuth = config_json['hillshade_azimuth']
                    if type(hillshade_azimuth) in (int, float):
                        config.hillshade_azimuth = hillshade_azimuth
                    else:
                        print(
                            f'Option hillshade_azimuth in config file "{str(hillshade_azimuth)}" is invalid: must be a number. Defaulting to {config.hillshade_azimuth}.')
                if 'hillshade_altitude' in config_json:
                    hillshade_altitude = config_json['hillshade_altitude']
                    if type(hillshade_altitude) in (int, float):
                        config.hillshade_altitude = hillshade_altitude
                    else:
                        print(
                            f'Option hillshade_altitude in config file "{str(hillshade_altitude)}" is invalid: must be a number. Defaulting to {config.hillshade_altitude}.')
                if 'verbose' in config_json:
                    verbose = config_json['verbose']
                    if type(verbose) == bool:
                        config.verbose = verbose
                    else:
                        print(
                            f'Option verbose in config file "{str(verbose)}" is invalid: must be a boolean. Defaulting to {config.verbose}.')
        except FileNotFoundError:
            print(f'No config file "{fname}" found. Using default values.')
            config = Config.default_config()
        except Exception as err:
            print(
                f'Error while reading config file "{fname}". Using default config values. Exception stacktrace:')
            print(traceback.format_exc())
            config = Config.default_config()
        
        return config
