#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include "argparse.hpp"
#include "dem.h"
#include "Cell.h"
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

typedef std::vector<Cell> NodeVector;
typedef std::priority_queue<Cell, NodeVector, Cell::Greater> PriorityQueue;

void my_InitPriorityQue_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Cell>& traceQueue,
    PriorityQueue& priorityQueue)
{
	int width=dem.Get_NX();
	int height=dem.Get_NY();
	Cell tmpNode;
	int iRow, iCol;
	// push border cells into the PQ
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col += width - 1)
        {
            tmpNode.row = row;
            tmpNode.col = col;
            tmpNode.spill_elevation = dem.asFloat(row, col);
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
            tmpNode.spill_elevation = dem.asFloat(row, col);
            priorityQueue.push(tmpNode);
            flag.set_true(row, col);
        }
    }
}

void my_ProcessPit_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Cell>& depressionQueue,
    queue<Cell>& traceQueue,
    PriorityQueue& priorityQueue,
    Sinkhole& sinkhole)
{
	int neighbor_row, neighbor_col,i;
	float neighbor_elevation;
	Cell neighbor_node;
	Cell current_node;
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
			if (neighbor_elevation > current_node.spill_elevation) 
			{
                // slope cell
				neighbor_node.row = neighbor_row;
				neighbor_node.col = neighbor_col;
				neighbor_node.spill_elevation = neighbor_elevation;				
				flag.set_true(neighbor_row,neighbor_col);
				traceQueue.push(neighbor_node);
			}
            else
            {
                // depression cell
                flag.set_true(neighbor_row,neighbor_col);
				if (!dem.is_NoData(neighbor_row, neighbor_col))
				{
                	dem.set_value(neighbor_row, neighbor_col, current_node.spill_elevation);
				}
                neighbor_node.row = neighbor_row;
                neighbor_node.col = neighbor_col;
                neighbor_node.spill_elevation = current_node.spill_elevation;
                depressionQueue.push(neighbor_node);

                sinkhole.update(dem, neighbor_row, neighbor_col, current_node.spill_elevation);
            }
		}
	}
}

void my_ProcessTraceQue_onepass(CDEM& dem,
    BitArray2d& flag,
    queue<Cell>& traceQueue,
    PriorityQueue& priorityQueue)
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
}

void fill_dem(CDEM& dem, std::vector<Sinkhole>& sinkholes, Settings& settings)
{
    // more or less directly copy FillDEM_Zhou_OnePass

    queue<Cell> traceQueue;
	queue<Cell> depressionQueue;
    
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
	float neighbor_elevation,spill_elevation;

	my_InitPriorityQue_onepass(dem,visited,traceQueue,priorityQueue);
	while (!priorityQueue.empty())
	{
		Cell cell = priorityQueue.top();
		priorityQueue.pop();
		row = cell.row;
		col = cell.col;
		spill_elevation = cell.spill_elevation;

		for (int i = 0; i < 8; i++)
		{

			neighbor_row = get_neighbor_row(i, row);
			neighbor_col = get_neighbor_col(i, col);

			if (visited.is_visited(neighbor_row,neighbor_col)) continue;
			neighbor_elevation = dem.asFloat(neighbor_row, neighbor_col);
			if (neighbor_elevation <= spill_elevation)
			{
				//depression cell
				if (!dem.is_NoData(neighbor_row, neighbor_col))
				{
					dem.set_value(neighbor_row, neighbor_col, spill_elevation);
				}
				visited.set_true(neighbor_row,neighbor_col);
				cell.row = neighbor_row;
				cell.col = neighbor_col;
				cell.spill_elevation = spill_elevation;
				depressionQueue.push(cell);
                Sinkhole sinkhole = Sinkhole();
                sinkhole.update(dem, neighbor_row, neighbor_col, spill_elevation);
				my_ProcessPit_onepass(dem, visited, depressionQueue, traceQueue, priorityQueue, sinkhole);
				if (sinkhole.max_depth >= settings.MIN_SINKHOLE_DEPTH && sinkhole.area >= settings.MIN_SINKHOLE_AREA)
				{
					sinkholes.push_back(sinkhole);
				}
			}
			else
			{
				//slope cell
				visited.set_true(neighbor_row,neighbor_col);
				cell.row = neighbor_row;
				cell.col = neighbor_col;
				cell.spill_elevation = neighbor_elevation;
				traceQueue.push(cell);
			}			
			my_ProcessTraceQue_onepass(dem,visited,traceQueue,priorityQueue);
		}
	}
    std::cout << "Finished filling DEM" << std::endl;
}

void handle_dem(string input_fname, optional<string> output_hillshade_fname, optional<string> output_sinkholes_fname, Settings& settings)
{
	CDEM dem;
    double geoTransofrmArgs[6];
    std::string wkt;
    bool read_result = readTIFF(input_fname.c_str(),
        GDALDataType::GDT_Float32, dem, geoTransofrmArgs, &wkt);
    if (!read_result)
    {
        std::cerr << "Error occurred while reading GeoTIFF file " << input_fname << ". Exiting." << endl;
        std::exit(1);
    }

	// Now we actually start processing the DEM

	// make the hillshade if output_hillshade_fname is specified
	if (output_hillshade_fname.has_value())
	{
		// TODO
	}

	// fill DEM and find sinkholes
	std::vector<Sinkhole> sinkholes;
    fill_dem(dem, sinkholes, settings);

	// write modified hillshade if output_hillshade_fname is specified
	if (output_hillshade_fname.has_value())
	{
		// TODO
	}

	// write sinkholes if output_sinkholes_fname is specified
	if (output_sinkholes_fname.has_value())
	{
		string fname = output_sinkholes_fname.value();
		
	}
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

	cout << "-oh argument: " << program.get<std::string>("-oh") << endl;
	return 0;

    if (!program.is_used("-oh") && !program.is_used("-os"))
    {
        std::cerr << "At least one output option must be specified. Exiting." << std::endl;
        std::cerr << program;
        std::exit(1);
    }

	Settings settings = Settings(); // TODO read from settings file if specified

    std::string input_fname = program.get<std::string>("-i");

    handle_dem(input_fname, program.get<std::string>("-oh"), program.get<std::string>("-os"), settings);

    return 0;
}