#include <string>

#include "settings.h"
#include "utils.h"

const std::string Settings::DEFAULT_COLORMAP = "rainbow_4_reverse";

Color Settings::depth_to_color(float depth)
{
    
    depth = std::max(depth, this->MIN_DEPTH_FOR_COLORMAP);
    depth = std::min(depth, this->MAX_DEPTH_FOR_COLORMAP);
    float normalized_depth_score = log(depth) - log(this->MIN_DEPTH_FOR_COLORMAP);
    normalized_depth_score /= (log(this->MAX_DEPTH_FOR_COLORMAP) - log(this->MIN_DEPTH_FOR_COLORMAP));
    int color_index = (int) std::round(normalized_depth_score * (COLORS_PER_CMAP - 1));
    this->COLORMAP[color_index];
}

std::string Settings::depth_to_gaiagps_color(float depth)
{
    // TODO
}