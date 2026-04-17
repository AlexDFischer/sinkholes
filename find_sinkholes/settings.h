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

#ifndef SETTINGS_H
#define SETTINGS_H

#include <stdexcept>
#include <vector>

#include "colors.h"

using namespace std;

class Settings
{
    public:
    const float MIN_SINKHOLE_DEPTH;
    const float MIN_SINKHOLE_AREA;
    const float MIN_DEPTH_FOR_COLORMAP;
    const float MAX_DEPTH_FOR_COLORMAP;
    const Colormap& COLORMAP;
    const string COLORMAP_NAME;
    const float HILLSHADE_Z_FACTOR;
    const float HILLSHADE_AZIMUTH;
    const float HILLSHADE_ALTITUDE;
    const std::vector<int> HILLSHADE_OVERVIEW_LEVELS;
    const int MAX_POINTS_PER_FILE;
    const bool VERBOSE;
    const float RESOLUTION;
    const std::vector<int> POINT_CLOUD_CLASSIFICATIONS;
    const string POINT_CLOUDS_DIR;
    const string DEMS_DIR;
    const string OUTPUT_DIR;
    const string SINKHOLES_QGIS_STYLE_FILE;
    string QGIS_PROJECT_FILE;
    const string QGIS_PYTHON_PATH;
    const string PYTHON_EXECUTABLE;
    const string SINKHOLES_QGIS_GROUP_NAME;
    const string HILLSHADE_QGIS_GROUP_NAME;
    const float NODATA_VALUE;

    Settings(
        float min_sinkhole_depth,
        float min_sinkhole_area,
        float min_depth_for_colormap,
        float max_depth_for_colormap,
        const string colormap,
        float hillshade_z_factor,
        float hillshade_azimuth,
        float hillshade_altitude,
        std::vector<int> hillshade_overview_levels,
        int max_points_per_file,
        bool verbose,
        float resolution,
        std::vector<int> point_cloud_classifications,
        string point_clouds_dir,
        string dems_dir,
        string output_dir,
        string sinkholes_qgis_style_file,
        string qgis_project_file,
        string qgis_python_path,
        string python_executable,
        string sinkholes_qgis_group_name,
        string hillshade_qgis_group_name,
        float nodata_value
    )
        : MIN_SINKHOLE_DEPTH(min_sinkhole_depth),
        MIN_SINKHOLE_AREA(min_sinkhole_area),
        MIN_DEPTH_FOR_COLORMAP(min_depth_for_colormap),
        MAX_DEPTH_FOR_COLORMAP(max_depth_for_colormap),
        COLORMAP(colormap_from_name(colormap)),
        COLORMAP_NAME(colormap),
        HILLSHADE_Z_FACTOR(hillshade_z_factor),
        HILLSHADE_AZIMUTH(hillshade_azimuth),
        HILLSHADE_ALTITUDE(hillshade_altitude),
        HILLSHADE_OVERVIEW_LEVELS(std::move(hillshade_overview_levels)),
        MAX_POINTS_PER_FILE(max_points_per_file),
        VERBOSE(verbose),
        RESOLUTION(resolution),
        POINT_CLOUD_CLASSIFICATIONS(std::move(point_cloud_classifications)),
        POINT_CLOUDS_DIR(std::move(point_clouds_dir)),
        DEMS_DIR(std::move(dems_dir)),
        OUTPUT_DIR(std::move(output_dir)),
        SINKHOLES_QGIS_STYLE_FILE(std::move(sinkholes_qgis_style_file)),
        QGIS_PROJECT_FILE(std::move(qgis_project_file)),
        QGIS_PYTHON_PATH(std::move(qgis_python_path)),
        PYTHON_EXECUTABLE(std::move(python_executable)),
        SINKHOLES_QGIS_GROUP_NAME(std::move(sinkholes_qgis_group_name)),
        HILLSHADE_QGIS_GROUP_NAME(std::move(hillshade_qgis_group_name)),
        NODATA_VALUE(nodata_value)
    {
    }

    static Settings from_json(const std::string& json_path);

    Color depth_to_color(float depth);

    string depth_to_gaiagps_color(float depth);
};

#endif
