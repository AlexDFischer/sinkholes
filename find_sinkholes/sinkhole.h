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

#ifndef SINKHOLE_H
#define SINKHOLE_H

#include <string>

#include "gdal_priv.h"
#include "ogr_spatialref.h"
#include "utils.h"

class Sinkhole
{
    public:
    int area;
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    float max_depth;
    float elevation;
    int x;
    int y;


    Sinkhole()
    : area(0),
    min_x(std::numeric_limits<int>::max()),
    max_x(std::numeric_limits<int>::min()),
    min_y(std::numeric_limits<int>::max()),
    max_y(std::numeric_limits<int>::min()),
    max_depth(0.0f),
    elevation(0.0f),
    x(0),
    y(0)
    {
    }

    void update(CDEM& dem, int row, int col, float spill_elevation);
    std::string info_string(float x_unit, float y_unit);

    Color get_color(Settings& settings);

    // Returns {latitude, longitude} in WGS84 decimal degrees
    std::pair<double, double> to_wgs84(const double* geo_transform, const std::string& wkt);
};

#endif