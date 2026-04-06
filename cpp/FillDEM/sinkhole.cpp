#include <limits>
#include <string>

#include "sinkhole.h"

std::string Sinkhole::info_string(float x_unit, float y_unit)
{
    std::string info = "Area: " + std::to_string(area) + " sq. " + x_unit + "*" + y_unit + "\n";
    info += "Min X: " + std::to_string(min_x) + " (" + std::to_string(min_x * x_unit) + " " + x_unit + ")\n";
    info += "Max X: " + std::to_string(max_x) + " (" + std::to_string(max_x * x_unit) + " " + x_unit + ")\n";
    info += "Min Y: " + std::to_string(min_y) + " (" + std::to_string(min_y * y_unit) + " " + y_unit + ")\n";
    info += "Max Y: " + std::to_string(max_y) + " (" + std::to_string(max_y * y_unit) + " " + y_unit + ")\n";
    info += "Max Depth: " + std::to_string(max_depth);
    return info;
}