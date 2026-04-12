#ifndef SETTINGS_H
#define SETTINGS_H

#include <vector>

#include "colors.h"

using namespace std;

class Settings
{
    public:
    static constexpr float DEFAULT_MIN_SINKHOLE_DEPTH = 0.5f;
    static constexpr float DEFAULT_MIN_SINKHOLE_AREA = 0.0f;
    static constexpr float DEFAULT_MIN_DEPTH_FOR_COLORMAP = 0.5f;
    static constexpr float DEFAULT_MAX_DEPTH_FOR_COLORMAP = 6.0f;
    static const string DEFAULT_COLORMAP;
    static constexpr float DEFAULT_HILLSHADE_Z_FACTOR = 1.0f;
    static constexpr float DEFAULT_HILLSHADE_AZIMUTH = 315.0f;
    static constexpr float DEFAULT_HILLSHADE_ALTITUDE = 30.0f;
    static constexpr int DEFAULT_MAX_POINTS_PER_FILE = -1;
    static constexpr bool DEFAULT_VERBOSE = true;
    static constexpr float DEFAULT_RESOLUTION = 1.0f;
    inline static const std::vector<int> DEFAULT_POINT_CLOUD_CLASSIFICATIONS = {1, 2, 7, 9, 21};
    static const string DEFAULT_POINT_CLOUDS_DIR;
    static const string DEFAULT_DEMS_DIR;
    static const string DEFAULT_OUTPUT_DIR;
    static constexpr float DEFAULT_NODATA_VALUE = -9999.0f;

    public:
    const float MIN_SINKHOLE_DEPTH;
    const float MIN_SINKHOLE_AREA;
    const float MIN_DEPTH_FOR_COLORMAP;
    const float MAX_DEPTH_FOR_COLORMAP;
    const Colormap& COLORMAP;
    const float HILLSHADE_Z_FACTOR;
    const float HILLSHADE_AZIMUTH;
    const float HILLSHADE_ALTITUDE;
    const int MAX_POINTS_PER_FILE;
    const bool VERBOSE;
    const float RESOLUTION;
    const std::vector<int> POINT_CLOUD_CLASSIFICATIONS;
    const string POINT_CLOUDS_DIR;
    const string DEMS_DIR;
    const string OUTPUT_DIR;
    const float NODATA_VALUE;

    Settings(
        float min_sinkhole_depth = DEFAULT_MIN_SINKHOLE_DEPTH,
        float min_sinkhole_area = DEFAULT_MIN_SINKHOLE_AREA,
        float min_depth_for_colormap = DEFAULT_MIN_DEPTH_FOR_COLORMAP,
        float max_depth_for_colormap = DEFAULT_MAX_DEPTH_FOR_COLORMAP,
        const string colormap = DEFAULT_COLORMAP,
        float hillshade_z_factor = DEFAULT_HILLSHADE_Z_FACTOR,
        float hillshade_azimuth = DEFAULT_HILLSHADE_AZIMUTH,
        float hillshade_altitude = DEFAULT_HILLSHADE_ALTITUDE,
        int max_points_per_file = DEFAULT_MAX_POINTS_PER_FILE,
        bool verbose = DEFAULT_VERBOSE,
        float resolution = DEFAULT_RESOLUTION,
        std::vector<int> point_cloud_classifications = DEFAULT_POINT_CLOUD_CLASSIFICATIONS,
        string point_clouds_dir = DEFAULT_POINT_CLOUDS_DIR,
        string dems_dir = DEFAULT_DEMS_DIR,
        string output_dir = DEFAULT_OUTPUT_DIR,
        float nodata_value = DEFAULT_NODATA_VALUE
    )
        : MIN_SINKHOLE_DEPTH(min_sinkhole_depth),
        MIN_SINKHOLE_AREA(min_sinkhole_area),
        MIN_DEPTH_FOR_COLORMAP(min_depth_for_colormap),
        MAX_DEPTH_FOR_COLORMAP(max_depth_for_colormap),
        COLORMAP(rainbow_4_reverse_colormap), // TODO function that initializes here from string
        HILLSHADE_Z_FACTOR(hillshade_z_factor),
        HILLSHADE_AZIMUTH(hillshade_azimuth),
        HILLSHADE_ALTITUDE(hillshade_altitude),
        MAX_POINTS_PER_FILE(max_points_per_file),
        VERBOSE(verbose),
        RESOLUTION(resolution),
        POINT_CLOUD_CLASSIFICATIONS(std::move(point_cloud_classifications)),
        POINT_CLOUDS_DIR(std::move(point_clouds_dir)),
        DEMS_DIR(std::move(dems_dir)),
        OUTPUT_DIR(std::move(output_dir)),
        NODATA_VALUE(nodata_value)
    {
    }

    static Settings from_json(const std::string& json_path);

    Color depth_to_color(float depth);

    string depth_to_gaiagps_color(float depth);
};

#endif