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

/**
 * Modified version of Guiyun Zhou, Zhongxuan Sun, and Suhua Fu's one-pass variant of their fast priotity-flood algorithm.
 * See https://github.com/zhouguiyun-uestc/FillDEM and https://www.sciencedirect.com/science/article/abs/pii/S0098300416300553
 * This modified file is distributed under GPL v3. The original MIT-licensed portions remain subject to the MIT license, reproduced below:

MIT License

Copyright (c) 2020 zhouguiyun-uestc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include "dem.h"
#include "Cell.h"
#include "utils.h"
#include <time.h>
#include <list>
#include <stack>
#include <unordered_map>
using namespace std;

typedef std::vector<Cell> NodeVector;
typedef std::priority_queue<Cell, NodeVector, Cell::Greater> PriorityQueue;
void InitPriorityQue_onepass(CDEM& dem, BitArray2d& flag, queue<Cell>& traceQueue, PriorityQueue& priorityQueue,int& percentFive)
{
	int width=dem.Get_NX();
	int height=dem.Get_NY();
	int validElementsCount = 0;
	Cell tmpNode;
	int iRow, iCol;
	// push border cells into the PQ
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (!dem.is_NoData(row, col))
			{
				validElementsCount++;
				for (int i = 0; i < 8; i++)
				{

					iRow = get_neighbor_row(i, row);
					iCol = get_neighbor_col(i, col);
					if (!dem.is_InGrid(iRow, iCol) || dem.is_NoData(iRow, iCol))
					{
						tmpNode.col = col;
						tmpNode.row = row;
						tmpNode.spill_elevation = dem.asFloat(row, col);
						priorityQueue.push(tmpNode);

						flag.set_true(row,col);
						break;
					}
				}
			}
			else{
				flag.set_true(row,col);
			}
		}
	}

	percentFive = validElementsCount / 20;
}
void ProcessTraceQue_onepass(CDEM& dem,BitArray2d& flag, queue<Cell>& traceQueue, PriorityQueue& priorityQueue,int& count, int percentFive)
{
	int iRow, iCol,i;
	float iSpill;
	Cell N,node,headNode;
	int width=dem.Get_NX();
	int height=dem.Get_NY();	
	int total=0,nPSC=0;
	bool bInPQ=false;
	bool isBoundary;
	int j,jRow,jCol;
	while (!traceQueue.empty())
	{
		node = traceQueue.front();
		traceQueue.pop();
		total++;
		if ((count+total) % percentFive == 0)
		{
			std::cout<<"Progress:"<<(count+total) / percentFive * 5 <<"%\r";
		}
		bInPQ=false;
 		for (i = 0; i < 8; i++)
		{
			iRow = get_neighbor_row(i, node.row);
			iCol = get_neighbor_col(i, node.col);
			if (flag.is_visited_skip_boundary_check(iRow,iCol)) continue;		
			
			iSpill = dem.asFloat(iRow, iCol);
			
			if (iSpill <= node.spill_elevation) 	{
				if (!bInPQ) {
					//decide  whether (iRow, iCol) is a true border cell
					isBoundary=true;
					for (j = 0; j < 8; j++)
					{
						jRow = get_neighbor_row(j, iRow);
						jCol = get_neighbor_col(j, iCol);
						if (flag.is_visited_skip_boundary_check(jRow,jCol) && dem.asFloat(jRow,jCol)<iSpill)
						{
							isBoundary=false;
							break;
						}
					}
					if (isBoundary) {
						priorityQueue.push(node);
						bInPQ=true;
						nPSC++;
					}
				}
				continue; 
			}
			//otherwise
			//N is unprocessed and N is higher than C
			N.col = iCol;
			N.row = iRow;
			N.spill_elevation = iSpill;
			traceQueue.push(N);
			flag.set_true(iRow,iCol);		
		}
	}
	count+=total-nPSC;
}

void ProcessPit_onepass(CDEM& dem, BitArray2d& flag, queue<Cell>& depressionQue,queue<Cell>& traceQueue,PriorityQueue& priorityQueue,int& count, int percentFive)
{
	int iRow, iCol,i;
	float iSpill;
	Cell N;
	Cell node;
	int width=dem.Get_NX();
	int height=dem.Get_NY();
	while (!depressionQue.empty())
	{
		node= depressionQue.front();
		depressionQue.pop();
		count++;
		if (count % percentFive == 0)
		{
			std::cout<<"Progress:"<<count / percentFive * 5 <<"%\r";
		}
		for (i = 0; i < 8; i++)
		{
			iRow = get_neighbor_row(i, node.row);
			iCol = get_neighbor_col(i,  node.col);

			if (flag.is_visited_skip_boundary_check(iRow,iCol)) continue;		
			iSpill = dem.asFloat(iRow, iCol);
			if (iSpill > node.spill_elevation) 
			{ //slope cell
				N.row = iRow;
				N.col = iCol;
				N.spill_elevation = iSpill;				
				flag.set_true(iRow,iCol);
				traceQueue.push(N);
				continue;
			}

			//depression cell
			flag.set_true(iRow,iCol);
			dem.set_value(iRow, iCol, node.spill_elevation);
			N.row = iRow;
			N.col = iCol;
			N.spill_elevation = node.spill_elevation;
			depressionQue.push(N);
		}
	}
}

void FillDEM_Zhou_OnePass(char* inputFile, char* outputFilledPath)
{
	queue<Cell> traceQueue;
	queue<Cell> depressionQue;

	//read float-type DEM
	CDEM dem;
	double geoTransformArgs[6];
	std::cout<<"Reading tiff files..."<<endl;
	if (!readTIFF(inputFile, GDALDataType::GDT_Float32, dem, geoTransformArgs))
	{
		printf("Error occurred while reading GeoTIFF file!\n");
		return;
	}	
	
	std::cout<<"Finish reading data"<<endl;

	time_t timeStart, timeEnd;
	int width = dem.Get_NX();
	int height = dem.Get_NY();
	
	timeStart = time(NULL);
	std::cout<<"Using the one-pass implementation of the proposed variant to fill DEM"<<endl;


	BitArray2d flag;
	if (!flag.Init(width,height)) {
		printf("Failed to allocate memory!\n");
		return;
	}

	PriorityQueue priorityQueue;
	int percentFive;
	int count = 0,potentialSpillCount=0;
	int iRow, iCol, row,col;
	float iSpill,spill;

	InitPriorityQue_onepass(dem,flag,traceQueue,priorityQueue,percentFive);
	while (!priorityQueue.empty())
	{
		Cell tmpNode = priorityQueue.top();
		priorityQueue.pop();
		count++;
		if (count % percentFive == 0)
		{
			std::cout<<"Progress:"<<count / percentFive * 5 <<"%\r";
		}
		row = tmpNode.row;
		col = tmpNode.col;
		spill = tmpNode.spill_elevation;

		for (int i = 0; i < 8; i++)
		{

			iRow = get_neighbor_row(i, row);
			iCol = get_neighbor_col(i, col);

			if (flag.is_visited(iRow,iCol)) continue;
			iSpill = dem.asFloat(iRow, iCol);
			if (iSpill <= spill)
			{
				//depression cell
				dem.set_value(iRow, iCol, spill);
				flag.set_true(iRow,iCol);
				tmpNode.row = iRow;
				tmpNode.col = iCol;
				tmpNode.spill_elevation = spill;
				depressionQue.push(tmpNode);
				ProcessPit_onepass(dem,flag,depressionQue,traceQueue,priorityQueue,count,percentFive);
			}
			else
			{
				//slope cell
				flag.set_true(iRow,iCol);
				tmpNode.row = iRow;
				tmpNode.col = iCol;
				tmpNode.spill_elevation = iSpill;
				traceQueue.push(tmpNode);
			}			
			ProcessTraceQue_onepass(dem,flag,traceQueue,priorityQueue,count,percentFive);
		}
	}
	timeEnd = time(NULL);
	double consumeTime = difftime(timeEnd, timeStart);
	std::cout<<"Time used:"<<consumeTime<<" seconds"<<endl;

	//����ͳ����
	double min, max, mean, stdDev;
	calculateStatistics(dem, &min, &max, &mean, &stdDev);
	CreateGeoTIFF(outputFilledPath, dem.Get_NY(), dem.Get_NX(), 
	(void *)dem.getDEMdata(),GDALDataType::GDT_Float32, geoTransformArgs,
	&min, &max, &mean, &stdDev, -9999);
	return;
}
