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