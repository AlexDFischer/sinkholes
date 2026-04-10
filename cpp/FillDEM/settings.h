#ifndef SETTINGS_H  
#define SETTINGS_H  

#include <string>

#include "colors.h"

using namespace std;

class Settings
{
    public:
    static constexpr float DEFAULT_MIN_SINKHOLE_DEPTH = 0.5;
    static constexpr float DEFAULT_MIN_SINKHOLE_AREA = 0.0;
    static constexpr float DEFAULT_MIN_DEPTH_FOR_COLORMAP = 0.5;
    static constexpr float DEFAULT_MAX_DEPTH_FOR_COLORMAP = 6.0;
    static const string DEFAULT_COLORMAP;
    static constexpr float DEFAULT_HILLSHADE_Z_FACTOR = 1.0;
    static constexpr float DEFAULT_HILLSHADE_AZIMUTH = 315.0;
    static constexpr float DEFAULT_HILLSHADE_ALTITUDE = 45.0;
    static constexpr int DEFAULT_MAX_POINTS_PER_FILE = -1;
    static constexpr bool DEFAULT_VERBOSE = true;


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
        bool verbose = DEFAULT_VERBOSE
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
        VERBOSE(verbose)
    {

    }

    Color depth_to_color(float depth);

    string depth_to_gaiagps_color(float depth);
};

#endif