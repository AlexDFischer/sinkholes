#ifndef SINKHOLE_H
#define SINKHOLE_H

#include <string>

class Sinkhole
{
    public:
    int area;
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    float max_depth;



    Sinkhole()
    : area(0),
    min_x(std::numeric_limits<int>::max()),
    max_x(std::numeric_limits<int>::min()),
    min_y(std::numeric_limits<int>::max()),
    max_y(std::numeric_limits<int>::min()),
    max_depth(0.0f)
    {
    }

    std::string info_string(float x_unit, float y_unit);
};

#endif