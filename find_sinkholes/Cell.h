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

#ifndef CELL_HEAD_H
#define CELL_HEAD_H
#include <functional>
class Cell
{
public:
	int row;
	int col;
	float spill_elevation;
	
	Cell()
	{
		row = 0;
		col = 0;
		spill_elevation = -9999.0;
	}

	struct Greater
	{
		bool operator()(const Cell n1, const Cell n2) const
		{
			return n1.spill_elevation > n2.spill_elevation;
		}
	};

	bool operator==(const Cell& a)
	{
		return (this->col == a.col) && (this->row == a.row);
	}
	bool operator!=(const Cell& a)
	{
		return (this->col != a.col) || (this->row != a.row);
	}
	bool operator<(const Cell& a)
	{
		return this->spill_elevation < a.spill_elevation;
	}
	bool operator>(const Cell& a)
	{
		return this->spill_elevation > a.spill_elevation;
	}
	bool operator>=(const Cell& a)
	{
		return this->spill_elevation >= a.spill_elevation;
	}
	bool operator<=(const Cell& a)
	{
		return this->spill_elevation <= a.spill_elevation;
	}
};

#endif