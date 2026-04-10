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
    int x;
    int y;


    Sinkhole()
    : area(0),
    min_x(std::numeric_limits<int>::max()),
    max_x(std::numeric_limits<int>::min()),
    min_y(std::numeric_limits<int>::max()),
    max_y(std::numeric_limits<int>::min()),
    max_depth(0.0f),
    x(0),
    y(0)
    {
    }

    void update(CDEM& dem, int row, int col, int spill_elevation);
    std::string info_string(float x_unit, float y_unit);

    Color get_color(Settings& settings);

    std::string to_wgs84(const double* geo_transform, const std::string& wkt);
};

#endif