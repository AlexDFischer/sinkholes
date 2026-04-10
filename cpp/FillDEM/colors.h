#ifndef COLORS_H 
#define COLORS_H

#include <cstdint>
#include <string>

using namespace std;

#define COLORS_PER_CMAP 256

struct Color
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

string color_to_hex(const Color& color);

typedef Color Colormap[COLORS_PER_CMAP];

extern Colormap plasma_reverse_colormap;

extern Colormap rainbow_4_reverse_colormap;

extern Colormap viridis_reverse_colormap;

extern Colormap spring_reverse_colormap;

extern Colormap winter_reverse_colormap;

extern Colormap cool_colormap;

extern Colormap gist_rainbow_colormap;

#endif