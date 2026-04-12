import json

class Settings:
    DEFAULT_VERBOSE = True
    DEFAULT_RESOLUTION = 1.0
    DEFAULT_POINT_CLOUD_CLASSIFICATIONS = [1, # Processed, but unclassified
                                           2, # Bare earth
                                           7, # Low noise
                                           9, # Water
                                           21 # Snow
                                           ]
    DEFAULT_POINT_CLOUDS_DIR = 'point_clouds'
    DEFAULT_DEMS_DIR = 'dems'
    DEFAULT_OUTPUT_DIR = 'output'
    DEFAULT_NODATA_VALUE = -9999.0

    def __init__(self,
                 verbose: bool=DEFAULT_VERBOSE,
                 resolution: float=1.0,
                 point_cloud_classifications: list[int]=[1, 2, 7],
                 point_clouds_dir: str=DEFAULT_POINT_CLOUDS_DIR,
                 dems_dir: str=DEFAULT_DEMS_DIR,
                 output_dir: str=DEFAULT_OUTPUT_DIR,
                 nodata_value: float=DEFAULT_NODATA_VALUE
                 ):
        self.verbose = verbose
        self.resolution = resolution
        self.point_cloud_classifications = point_cloud_classifications
        self.point_clouds_dir = point_clouds_dir
        self.dems_dir = dems_dir
        self.output_dir = output_dir
        self.nodata_value = nodata_value
    
    @classmethod
    def from_json(cls, json_file: str):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return cls(**data)