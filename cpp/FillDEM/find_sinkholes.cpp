#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include "argparse.hpp"
#include "dem.h"
#include "Node.h"
#include "settings.h"
#include "sinkhole.h"
#include "utils.h"
#include <time.h>
#include <list>
#include <unordered_map>


using namespace std;
using std::cout;
using std::endl;
using std::string;
using std::getline;
using std::fstream;
using std::ifstream;
using std::priority_queue;
using std::binary_function;

typedef std::vector<Node> NodeVector;
typedef std::priority_queue<Node, NodeVector, Node::Greater> PriorityQueue;

void my_InitPriorityQue_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Node>& traceQueue,
    PriorityQueue& priorityQueue)
{
	int width=dem.Get_NX();
	int height=dem.Get_NY();
	Node tmpNode;
	int iRow, iCol;
	// push border cells into the PQ
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col += width - 1)
        {
            tmpNode.row = row;
            tmpNode.col = col;
            tmpNode.spill = dem.asFloat(row, col);
            priorityQueue.push(tmpNode);
            flag.set_true(row, col);
        }
    }

    for (int col = 0; col < width; col++)
    {
        for (int row = 0; row < height; row += height - 1)
        {
            tmpNode.row = row;
            tmpNode.col = col;
            tmpNode.spill = dem.asFloat(row, col);
            priorityQueue.push(tmpNode);
            flag.set_true(row, col);
        }
    }
}

void update_sinkhole(Sinkhole& sinkhole, CDEM& dem, int row, int col, int spill_elevation)
{
    sinkhole.area += 1;
    sinkhole.min_x = min(sinkhole.min_x, col);
    sinkhole.max_x = max(sinkhole.max_x, col);
    sinkhole.min_y = min(sinkhole.min_y, row);
    sinkhole.max_y = max(sinkhole.max_y, row);
    sinkhole.max_depth = max(sinkhole.max_depth,
        dem.is_NoData(row, col) ? 0.0f : spill_elevation - dem.asFloat(row, col));
}

void my_ProcessPit_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Node>& depressionQueue,
    queue<Node>& traceQueue,
    PriorityQueue& priorityQueue,
    Sinkhole& sinkhole)
{
	int neighbor_row, neighbor_col,i;
	float neighbor_elevation;
	Node neighbor_node;
	Node current_node;
	int width=dem.Get_NX();
	int height=dem.Get_NY();
	while (!depressionQueue.empty())
	{
		current_node = depressionQueue.front();
		depressionQueue.pop();

		for (i = 0; i < 8; i++)
		{
			neighbor_row = get_neighbor_row(i, current_node.row);
			neighbor_col = get_neighbor_col(i,  current_node.col);

			if (flag.is_visited_skip_boundary_check(neighbor_row,neighbor_col))
            {
                continue;		
            }

			neighbor_elevation = dem.asFloat(neighbor_row, neighbor_col);
			if (neighbor_elevation > current_node.spill) 
			{
                // slope cell
				neighbor_node.row = neighbor_row;
				neighbor_node.col = neighbor_col;
				neighbor_node.spill = neighbor_elevation;				
				flag.set_true(neighbor_row,neighbor_col);
				traceQueue.push(neighbor_node);
			}
            else
            {
                // depression cell
                flag.set_true(neighbor_row,neighbor_col);
                dem.set_value(neighbor_row, neighbor_col, current_node.spill);
                neighbor_node.row = neighbor_row;
                neighbor_node.col = neighbor_col;
                neighbor_node.spill = current_node.spill;
                depressionQueue.push(neighbor_node);

                update_sinkhole(sinkhole, dem, neighbor_row, neighbor_col, current_node.spill);
            }
		}
	}
}

void my_ProcessTraceQue_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Node>& traceQueue,
    PriorityQueue& priorityQueue)
{
	int iRow, iCol,i;
	float iSpill;
	Node N,node,headNode;
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
		bInPQ=false;
 		for (i = 0; i < 8; i++)
		{
			iRow = get_neighbor_row(i, node.row);
			iCol = get_neighbor_col(i, node.col);
			if (flag.is_visited_skip_boundary_check(iRow,iCol)) continue;		
			
			iSpill = dem.asFloat(iRow, iCol);
			
			if (iSpill <= node.spill) 	{
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
			N.spill = iSpill;
			traceQueue.push(N);
			flag.set_true(iRow,iCol);		
		}
	}
}

void filled_difference_direct(CDEM& dem)
{
    // more or less directly copy FillDEM_Zhou_OnePass

    vector<Sinkhole> sinkholes;
	queue<Node> traceQueue;
	queue<Node> depressionQueue;
    
    int width = dem.Get_NX();
	int height = dem.Get_NY();

	BitArray2d visited;
	if (!visited.Init(width,height)) {
		printf("Failed to allocate memory!\n");
		return;
	}

	PriorityQueue priorityQueue;
	int percentFive;
	int neighbor_row, neighbor_col, row,col;
	float neighbor_elevation,spill;

	my_InitPriorityQue_onepass(dem,visited,traceQueue,priorityQueue);
	while (!priorityQueue.empty())
	{
		Node tmpNode = priorityQueue.top();
		priorityQueue.pop();
		row = tmpNode.row;
		col = tmpNode.col;
		spill = tmpNode.spill;

		for (int i = 0; i < 8; i++)
		{

			neighbor_row = get_neighbor_row(i, row);
			neighbor_col = get_neighbor_col(i, col);

			if (visited.is_visited(neighbor_row,neighbor_col)) continue;
			neighbor_elevation = dem.asFloat(neighbor_row, neighbor_col);
			if (neighbor_elevation <= spill)
			{
				//depression cell
				dem.set_value(neighbor_row, neighbor_col, spill);
				visited.set_true(neighbor_row,neighbor_col);
				tmpNode.row = neighbor_row;
				tmpNode.col = neighbor_col;
				tmpNode.spill = spill;
				depressionQueue.push(tmpNode);
                Sinkhole& sinkhole = sinkholes.emplace_back();
                update_sinkhole(sinkhole, dem, neighbor_row, neighbor_col, spill);
				my_ProcessPit_onepass(dem, visited, depressionQueue, traceQueue, priorityQueue, sinkhole);
			}
			else
			{
				//slope cell
				visited.set_true(neighbor_row,neighbor_col);
				tmpNode.row = neighbor_row;
				tmpNode.col = neighbor_col;
				tmpNode.spill = neighbor_elevation;
				traceQueue.push(tmpNode);
			}			
			my_ProcessTraceQue_onepass(dem,visited,traceQueue,priorityQueue);
		}
	}
    std::cout << "Finished filling DEM" << std::endl;
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("find_sinkholes", "0.1.0");
    program.add_argument("-i", "--input")
        .help("Input DEM to find sinkholes within.")
        .required();
    program.add_argument("-oh", "--output-hillshade")
        .help("Output hillshade raster .tif with sinkholes highlighted.");
    program.add_argument("-os", "--output-sinkholes")
        .help("Output .geojson file with sinkholes.");
    
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std:cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    if (!program.is_used("-oh") && !program.is_used("-os"))
    {
        std::cerr << "At least one output option must be specified. Exiting." << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string input_fname = program.get<std::string>("-i");

    CDEM dem;
    double geoTransofrmArgs[6];
    bool read_result = readTIFF(input_fname.c_str(),
        GDALDataType::GDT_Float32, dem, geoTransofrmArgs);
    if (!read_result)
    {
        std::cerr << "Error occurred while reading GeoTIFF file " << input_fname << ". Exiting." << endl;
        std::exit(1);
    }

    filled_difference_direct(dem);

	double min, max, mean, stdDev;
	calculateStatistics(dem, &min, &max, &mean, &stdDev);

    std::string output_hillshade_fname = program.get<std::string>("-oh");
    const char* output_hillshade_fname_cstr = output_hillshade_fname.c_str();
    CreateGeoTIFF(output_hillshade_fname_cstr,
        dem.Get_NY(),
        dem.Get_NX(),
        dem.getDEMdata(),
        GDALDataType::GDT_Float32,
        geoTransofrmArgs,
        &min,
        &max,
        &mean,
        &stdDev,
        NO_DATA_VALUE);
    
    cout << "Wrote filled dem to " << output_hillshade_fname << endl;
    cout << "geoTransformArgs:" << endl;
    for (int i = 0; i < 6; i++)
    {
        cout << geoTransofrmArgs[i] << " ";
    }
    cout << endl;
    // Construct difference DEM

    return 0;
}