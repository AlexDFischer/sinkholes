#include <fstream>
#include <string>

#include "json.hpp"
#include "settings.h"
#include "utils.h"

using namespace std;

const std::string Settings::DEFAULT_COLORMAP        = "rainbow_4_reverse";
const std::string Settings::DEFAULT_POINT_CLOUDS_DIR = "point_clouds";
const std::string Settings::DEFAULT_DEMS_DIR         = "dems";
const std::string Settings::DEFAULT_OUTPUT_DIR       = "output";

Settings Settings::from_json(const std::string& json_path)
{
    std::ifstream f(json_path);
    if (!f)
        throw std::runtime_error("Could not open settings file: " + json_path);
    nlohmann::json j;
    f >> j;

    return Settings(
        j.value("min_sinkhole_depth",            DEFAULT_MIN_SINKHOLE_DEPTH),
        j.value("min_sinkhole_area",             DEFAULT_MIN_SINKHOLE_AREA),
        j.value("min_depth_for_colormap",        DEFAULT_MIN_DEPTH_FOR_COLORMAP),
        j.value("max_depth_for_colormap",        DEFAULT_MAX_DEPTH_FOR_COLORMAP),
        j.value("colormap",                      DEFAULT_COLORMAP),
        j.value("hillshade_z_factor",            DEFAULT_HILLSHADE_Z_FACTOR),
        j.value("hillshade_azimuth",             DEFAULT_HILLSHADE_AZIMUTH),
        j.value("hillshade_altitude",            DEFAULT_HILLSHADE_ALTITUDE),
        j.value("max_points_per_file",           DEFAULT_MAX_POINTS_PER_FILE),
        j.value("verbose",                       DEFAULT_VERBOSE),
        j.value("resolution",                    DEFAULT_RESOLUTION),
        j.value("point_cloud_classifications",   DEFAULT_POINT_CLOUD_CLASSIFICATIONS),
        j.value("point_clouds_dir",              DEFAULT_POINT_CLOUDS_DIR),
        j.value("dems_dir",                      DEFAULT_DEMS_DIR),
        j.value("output_dir",                    DEFAULT_OUTPUT_DIR),
        j.value("nodata_value",                  DEFAULT_NODATA_VALUE)
    );
}

array<string, 6> gaiagps_pin_filenames = {
    "red-pin.png",
    "orange-pin.png",
    "yellow-pin.png",
    "green-pin.png",
    "blue-pin-down.png",
    "purple-pin.png"
};

Color Settings::depth_to_color(float depth)
{
    depth = std::max(depth, this->MIN_DEPTH_FOR_COLORMAP);
    depth = std::min(depth, this->MAX_DEPTH_FOR_COLORMAP);
    float normalized_depth_score = log(depth) - log(this->MIN_DEPTH_FOR_COLORMAP);
    normalized_depth_score /= (log(this->MAX_DEPTH_FOR_COLORMAP) - log(this->MIN_DEPTH_FOR_COLORMAP));
    int color_index = static_cast<int>(normalized_depth_score * COLORS_PER_CMAP);
    color_index = max(0, min(color_index, COLORS_PER_CMAP - 1));
    return this->COLORMAP[color_index];
}

std::string Settings::depth_to_gaiagps_color(float depth)
{
    depth = std::max(depth, this->MIN_DEPTH_FOR_COLORMAP);
    depth = std::min(depth, this->MAX_DEPTH_FOR_COLORMAP);
    float normalized_depth_score = log(depth) - log(this->MIN_DEPTH_FOR_COLORMAP);
    normalized_depth_score /= (log(this->MAX_DEPTH_FOR_COLORMAP) - log(this->MIN_DEPTH_FOR_COLORMAP));
    int color_index = static_cast<int>(normalized_depth_score * gaiagps_pin_filenames.size());
    color_index = max(0, min(color_index, static_cast<int>(gaiagps_pin_filenames.size()) - 1));
    return gaiagps_pin_filenames[color_index];
}