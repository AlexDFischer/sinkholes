#include <limits>
#include <string>

#include "dem.h"
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