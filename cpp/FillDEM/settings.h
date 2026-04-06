#include <string>

class Settings
{
    public:
    static constexpr float DEFAULT_RESOLUTION_X = 1.0;
    static constexpr float DEFAULT_RESOLUTION_Y = 1.0;
    static constexpr float DEFAULT_MIN_SINKHOLE_DEPTH = 0.5;
    static constexpr float DEFAULT_MIN_SINKHOLE_AREA = 0.0;
    static constexpr float DEFAULT_MIN_DEPTH_FOR_COLORMAP = 0.5;
    static constexpr float DEFAULT_MAX_DEPTH_FOR_COLORMAP = 6.0;
    static const std::string DEFAULT_POINT_COLORMAP;
    static const std::string DEFAULT_HILLSHADE_COLORMAP;
    static constexpr float DEFAULT_HILLSHADE_Z_FACTOR = 1.0;
    static constexpr float DEFAULT_HILLSHADE_AZIMUTH = 315.0;
    static constexpr float DEFAULT_HILLSHADE_ALTITUDE = 45.0;
    static constexpr bool DEFAULT_VERBOSE = true;

    public:
    const float RESOLUTION_X;
    const float RESOLUTION_Y;
    const float MIN_SINKHOLE_DEPTH;
    const float MIN_SINKHOLE_AREA;
    const float MIN_DEPTH_FOR_COLORMAP;
    const float MAX_DEPTH_FOR_COLORMAP;
    const std::string POINT_COLORMAP;
    const std::string HILLSHADE_COLORMAP;
    const float HILLSHADE_Z_FACTOR;
    const float HILLSHADE_AZIMUTH;
    const float HILLSHADE_ALTITUDE;
    const bool VERBOSE;

    Settings(
        float resolution_x = DEFAULT_RESOLUTION_X,
        float resolution_y = DEFAULT_RESOLUTION_Y,
        float min_sinkhole_depth = DEFAULT_MIN_SINKHOLE_DEPTH,
        float min_sinkhole_area = DEFAULT_MIN_SINKHOLE_AREA,
        float min_depth_for_colormap = DEFAULT_MIN_DEPTH_FOR_COLORMAP,
        float max_depth_for_colormap = DEFAULT_MAX_DEPTH_FOR_COLORMAP,
        const std::string& point_colormap_name = DEFAULT_POINT_COLORMAP,
        const std::string& hillshade_colormap_name = DEFAULT_HILLSHADE_COLORMAP,
        float hillshade_z_factor = DEFAULT_HILLSHADE_Z_FACTOR,
        float hillshade_azimuth = DEFAULT_HILLSHADE_AZIMUTH,
        float hillshade_altitude = DEFAULT_HILLSHADE_ALTITUDE,
        bool verbose = DEFAULT_VERBOSE
    )
        : RESOLUTION_X(resolution_x),
        RESOLUTION_Y(resolution_y),
        MIN_SINKHOLE_DEPTH(min_sinkhole_depth),
        MIN_SINKHOLE_AREA(min_sinkhole_area),
        MIN_DEPTH_FOR_COLORMAP(min_depth_for_colormap),
        MAX_DEPTH_FOR_COLORMAP(max_depth_for_colormap),
        POINT_COLORMAP(point_colormap_name),
        HILLSHADE_COLORMAP(hillshade_colormap_name),
        HILLSHADE_Z_FACTOR(hillshade_z_factor),
        HILLSHADE_AZIMUTH(hillshade_azimuth),
        HILLSHADE_ALTITUDE(hillshade_altitude),
        VERBOSE(verbose)
    {

    }
};