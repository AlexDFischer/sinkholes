#include <string>

#include "settings.h"
#include "utils.h"

using namespace std;

const std::string Settings::DEFAULT_COLORMAP = "rainbow_4_reverse";

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