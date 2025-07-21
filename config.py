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
