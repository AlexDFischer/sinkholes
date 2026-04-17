// Copyright (C) 2026 Alex Fischer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <fstream>
#include <string>

#include "json.hpp"
#include "settings.h"
#include "utils.h"

using namespace std;

Settings Settings::from_json(const std::string& json_path)
{
    std::ifstream f(json_path);
    if (!f)
        throw std::runtime_error("Could not open settings file: " + json_path);
    nlohmann::json j;
    f >> j;

    return Settings(
        j.at("min_sinkhole_depth")           .get<float>(),
        j.at("min_sinkhole_area")            .get<float>(),
        j.at("min_depth_for_colormap")       .get<float>(),
        j.at("max_depth_for_colormap")       .get<float>(),
        j.at("colormap")                     .get<std::string>(),
        j.at("hillshade_z_factor")           .get<float>(),
        j.at("hillshade_azimuth")            .get<float>(),
        j.at("hillshade_altitude")           .get<float>(),
        j.at("hillshade_overview_levels")    .get<std::vector<int>>(),
        j.at("max_points_per_file")          .get<int>(),
        j.at("verbose")                      .get<bool>(),
        j.at("resolution")                   .get<float>(),
        j.at("point_cloud_classifications")  .get<std::vector<int>>(),
        j.at("point_clouds_dir")             .get<std::string>(),
        j.at("dems_dir")                     .get<std::string>(),
        j.at("output_dir")                   .get<std::string>(),
        j.at("sinkholes_qgis_style_file")    .get<std::string>(),
        j.at("qgis_project_file")            .get<std::string>(),
        j.at("qgis_python_path")             .get<std::string>(),
        j.at("python_executable")            .get<std::string>(),
        j.at("sinkholes_qgis_group_name")    .get<std::string>(),
        j.at("hillshade_qgis_group_name")    .get<std::string>(),
        j.at("nodata_value")                 .get<float>()
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