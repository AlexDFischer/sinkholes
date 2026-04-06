#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include "dem.h"
#include "Node.h"
#include "utils.h"
#include <time.h>
#include <list>
#include <unordered_map>
using namespace std;

typedef std::vector<Cell> NodeVector;
typedef std::priority_queue<Cell, NodeVector, Cell::Greater> PriorityQueue;
void InitPriorityQue(CDEM& dem, BitArray2d& flag, BitArray2d& flag2, queue<Cell>& traceQueue, PriorityQueue& priorityQueue,int& percentFive)
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

						flag.SetFlags(row,col,flag2);
						break;
					}
				}
			}
			else{
				flag.SetFlags(row,col,flag2);
			}
		}
	}

	percentFive = validElementsCount / 20;
}
void ProcessTraceQue(CDEM& dem,BitArray2d& flag, BitArray2d& flag2,queue<Cell>& traceQueue, PriorityQueue& priorityQueue,int& count, int percentFive)
{
	int iRow, iCol,i;
	float iSpill;
	Cell N,node,headNode;
	int width=dem.Get_NX();
	int height=dem.Get_NY();	
	queue<Cell> traceQueue2(traceQueue);
	int total=0;
	while (!traceQueue.empty())
	{
		node = traceQueue.front();
		traceQueue.pop();
		total++;
		if ((count+total/2) % percentFive == 0)
		{
			std::cout<<"Progress:"<<(count+total/2) / percentFive * 5 <<"%\r";
		}

 		for (i = 0; i < 8; i++)
		{
			iRow = get_neighbor_row(i, node.row);
			iCol = get_neighbor_col(i, node.col);
			if (flag.is_visited_skip_boundary_check(iRow,iCol)) continue;		
			
			iSpill = dem.asFloat(iRow, iCol);
			
			if (iSpill <= node.spill_elevation) 
				continue;  

			//slope cell
			N.col = iCol;
			N.row = iRow;
			N.spill_elevation = iSpill;
			traceQueue.push(N);
			flag.set_true(iRow,iCol);		
		}
	}
	int nPSC=0;
	int count0=count;
	count+=total/2;
	total=0;
	bool bInPQ=false;
	while (!traceQueue2.empty())
	{
		node = traceQueue2.front();
		traceQueue2.pop();
		total++;
		if ((count+total/2) % percentFive == 0)
		{
			std::cout<<"Progress:"<<(count+total/2) / percentFive * 5 <<"%\r";
		}

		bInPQ=false;
		for (i = 0; i < 8; i++)
		{
			iRow = get_neighbor_row(i, node.row);
			iCol = get_neighbor_col(i, node.col);
			if (flag2.is_visited_skip_boundary_check(iRow,iCol)) continue;					

			if (flag.is_visited_skip_boundary_check(iRow,iCol)){
				N.col = iCol;
				N.row = iRow;
			    flag2.set_true(iRow,iCol);
				traceQueue2.push(N);
			}
			else {
				if (!bInPQ) {
					node.spill_elevation=dem.asFloat(node.row, node.col);
					priorityQueue.push(node);				
					bInPQ=true;
					nPSC++;
				}
			}
		}
	}	
	count=count0+total-nPSC;
}

void ProcessPit(CDEM& dem, BitArray2d& flag, BitArray2d& flag2,queue<Cell>& depressionQue,queue<Cell>& traceQueue,PriorityQueue& priorityQueue,int& count, int percentFive)
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
			{   //slope cell
				N.row = iRow;
				N.col = iCol;
				N.spill_elevation = iSpill;
				traceQueue.push(N);
				flag.set_true(iRow,iCol);
				flag2.set_true(iRow,iCol);
				continue;
			}

			//depressio cell
			flag.SetFlags(iRow,iCol,flag2);
			dem.set_value(iRow, iCol, node.spill_elevation);
			N.row = iRow;
			N.col = iCol;
			N.spill_elevation = node.spill_elevation;
			depressionQue.push(N);
		}
	}
}

void FillDEM_Zhou_TwoPass(char* inputFile, char* outputFilledPath)
{
	queue<Cell> traceQueue;//׷�ٶ���
	queue<Cell> depressionQue;//�ݵص��б�

	//��������
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
	std::cout<<"Using the two-pass implementation of the proposed variant to fill DEM"<<endl;


	BitArray2d flag;
	if (!flag.Init(width,height)) {
		printf("Out of memory!\n");
		return;
	}

	BitArray2d flag2;
	if (!flag2.Init(width,height)) {
		printf("Failed to allocate memory!\n");
		return;
	}

	PriorityQueue priorityQueue;
	int percentFive;
	int count = 0,potentialSpillCount=0;
	int iRow, iCol, row,col;
	float iSpill,spill;

	InitPriorityQue(dem,flag,flag2,traceQueue,priorityQueue,percentFive);
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

			iRow = Get_rowTo(i, row);
			iCol = get_neighbor_col(i, col);

			if (flag.is_visited(iRow,iCol)) continue;

			iSpill = dem.asFloat(iRow, iCol);
			if (iSpill <= spill)
			{
				//depression cell
				dem.set_value(iRow, iCol, spill);
				flag.SetFlags(iRow,iCol,flag2);
				tmpNode.row = iRow;
				tmpNode.col = iCol;
				tmpNode.spill_elevation = spill;
				depressionQue.push(tmpNode);
				ProcessPit(dem,flag,flag2,depressionQue,traceQueue,priorityQueue,count,percentFive);
			}
			else
			{
				//slope cell
				flag.SetFlags(iRow,iCol,flag2);
				tmpNode.row = iRow;
				tmpNode.col = iCol;
				tmpNode.spill_elevation = iSpill;
				traceQueue.push(tmpNode);
			}			
			ProcessTraceQue(dem,flag,flag2,traceQueue,priorityQueue,count,percentFive);
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
