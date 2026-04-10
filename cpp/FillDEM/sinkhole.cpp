#include <limits>
#include <string>

#include "dem.h"
#include "settings.h"
#include "sinkhole.h"

std::string Sinkhole::info_string(float x_unit, float y_unit)
{
    std::string info = "Area: " + std::to_string(area * x_unit * y_unit) + " m^2\n";
    info += "Min X: " + std::to_string(min_x) + "\n";
    info += "Max X: " + std::to_string(max_x) + "\n";
    info += "Min Y: " + std::to_string(min_y) + "\n";
    info += "Max Y: " + std::to_string(max_y) + "\n";
    info += "Max Depth: " + std::to_string(max_depth);
    return info;
}

void Sinkhole::update(CDEM& dem, int row, int col, int spill_elevation)
{
    this->area += 1;
    this->min_x = std::min(this->min_x, col);
    this->max_x = std::max(this->max_x, col);
    this->min_y = std::min(this->min_y, row);
    this->max_y = std::max(this->max_y, row);
    float cur_depth = spill_elevation - dem.asFloat(row, col);
    if (!dem.is_NoData(row, col) && cur_depth > this->max_depth)
    {
        this->max_depth = cur_depth;
        this->x = col;
        this->y = row;
    }
}

std::string Sinkhole::to_wgs84(const double* geo_transform, const std::string& wkt)
{
    // Convert pixel (col, row) to projected coordinates using the geotransform
    double proj_x = geo_transform[0] + x * geo_transform[1] + y * geo_transform[2];
    double proj_y = geo_transform[3] + x * geo_transform[4] + y * geo_transform[5];

    // Set up source CRS from the DEM's WKT
    OGRSpatialReference src_crs;
    src_crs.importFromWkt(wkt.c_str());

    // Set up target CRS (WGS84 geographic)
    OGRSpatialReference dst_crs;
    dst_crs.SetWellKnownGeogCS("WGS84");

    // Ensure axis order is (longitude, latitude) regardless of CRS definition
    dst_crs.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    OGRCoordinateTransformation* transform = OGRCreateCoordinateTransformation(&src_crs, &dst_crs);
    if (transform == nullptr)
    {
        return "";
    }

    double lon = proj_x, lat = proj_y;
    transform->Transform(1, &lon, &lat);
    OGRCoordinateTransformation::DestroyCT(transform);

    return std::to_string(lat) + ", " + std::to_string(lon);
}