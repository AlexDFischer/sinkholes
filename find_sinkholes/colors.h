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