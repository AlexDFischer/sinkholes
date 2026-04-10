#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "argparse.hpp"
#include "json.hpp"
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
    BitArray2d& visited,
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

			if (visited.is_visited_skip_boundary_check(neighbor_row,neighbor_col))
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
				visited.set_true(neighbor_row,neighbor_col);
				traceQueue.push(neighbor_node);
			}
            else
            {
                // depression cell
                sinkhole.update(dem, neighbor_row, neighbor_col, current_node.spill_elevation);

                visited.set_true(neighbor_row,neighbor_col);
				if (!dem.is_NoData(neighbor_row, neighbor_col))
				{
                	dem.set_value(neighbor_row, neighbor_col, current_node.spill_elevation);
				}
                neighbor_node.row = neighbor_row;
                neighbor_node.col = neighbor_col;
                neighbor_node.spill_elevation = current_node.spill_elevation;
                depressionQueue.push(neighbor_node);
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
		exit(1);
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
                Sinkhole sinkhole = Sinkhole();
				sinkhole.elevation = spill_elevation;
                sinkhole.update(dem, neighbor_row, neighbor_col, spill_elevation);

				if (!dem.is_NoData(neighbor_row, neighbor_col))
				{
					dem.set_value(neighbor_row, neighbor_col, spill_elevation);
				}
				visited.set_true(neighbor_row,neighbor_col);
				cell.row = neighbor_row;
				cell.col = neighbor_col;
				cell.spill_elevation = spill_elevation;
				depressionQueue.push(cell);

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
}

static std::string generate_uuid()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> hex_dist(0, 15);
    static std::uniform_int_distribution<> variant_dist(8, 11);

    std::ostringstream ss;
    ss << std::hex;
    for (int i = 0; i < 8;  i++) ss << hex_dist(gen);
    ss << "-";
    for (int i = 0; i < 4;  i++) ss << hex_dist(gen);
    ss << "-4"; // UUID version 4
    for (int i = 0; i < 3;  i++) ss << hex_dist(gen);
    ss << "-" << variant_dist(gen);
    for (int i = 0; i < 3;  i++) ss << hex_dist(gen);
    ss << "-";
    for (int i = 0; i < 12; i++) ss << hex_dist(gen);
    return ss.str();
}

static std::string uuid_to_hex(const std::string& uuid)
{
    std::string hex;
    for (char c : uuid)
        if (c != '-') hex += c;
    return hex;
}

static std::string current_datetime_str()
{
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    return buf;
}

// Returns pixels -> meters conversion for each axis, or {-1, -1} if CRS is geographic
static std::pair<double, double> get_pixel_size_meters(const double* geo_transform, const std::string& wkt)
{
    OGRSpatialReference crs;
    crs.importFromWkt(wkt.c_str());

    if (!crs.IsProjected())
        return {-1.0, -1.0};

    double linear_units = crs.GetLinearUnits(); // CRS units -> meters
    double pixel_width_m  = std::fabs(geo_transform[1]) * linear_units;
    double pixel_height_m = std::fabs(geo_transform[5]) * linear_units;
    return {pixel_width_m, pixel_height_m};
}

static std::string area_to_string(int area_cells, double pixel_width_m, double pixel_height_m)
{
    if (pixel_width_m < 0)
        return std::to_string(area_cells) + " cells";

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << area_cells * pixel_width_m * pixel_height_m << "m^2";
    return ss.str();
}

static void write_sinkholes_geojson_chunk(
    const std::string& output_fname,
    const std::vector<Sinkhole>& sinkholes,
    const double* geo_transform,
    const std::string& wkt,
    Settings& settings)
{
	if (sinkholes.empty())
	{
		if (settings.VERBOSE)
		{
			cout << "No sinkholes to export for " << output_fname << ", skipping file creation." << std::endl;
		}
		return;
	}

    using json = nlohmann::json;

    std::string folder_uuid = generate_uuid();
    std::string folder_uuid_hex = uuid_to_hex(folder_uuid);
    std::string now_str = current_datetime_str();
    auto [pixel_width_m, pixel_height_m] = get_pixel_size_meters(geo_transform, wkt);

    float min_depth = sinkholes[0].max_depth;
    float max_depth = sinkholes[0].max_depth;
    for (const Sinkhole& s : sinkholes)
    {
        min_depth = std::min(min_depth, s.max_depth);
        max_depth = std::max(max_depth, s.max_depth);
    }

    std::ostringstream notes_ss;
    notes_ss << std::fixed << std::setprecision(1)
             << sinkholes.size() << " sinkholes automatically detected by caves.science. "
             << "Depths range from " << min_depth << "m to " << max_depth << "m.";

    json output = {
        {"type", "FeatureCollection"},
        {"properties", {
            {"name", "caves.science automatic sinkholes"},
            {"updated_date", now_str},
            {"time_created", now_str},
            {"notes", notes_ss.str()}
        }},
        {"features", json::array()}
    };

    // Folder feature for Caltopo
    output["features"].push_back({
        {"geometry", nullptr},
        {"id", folder_uuid_hex},
        {"type", "Feature"},
        {"properties", {
            {"visible", true},
            {"title", "caves.science automatic sinkholes"},
            {"class", "Folder"},
            {"labelVisible", true}
        }}
    });

    for (const Sinkhole& s : sinkholes)
    {
        auto [lat, lon] = const_cast<Sinkhole&>(s).to_wgs84(geo_transform, wkt);

        std::ostringstream title_ss;
        title_ss << std::fixed << std::setprecision(1) << "sinkhole " << s.max_depth << "d";
        if (pixel_width_m >= 0)
        {
            double width_m  = (s.max_x - s.min_x) * pixel_width_m;
            double height_m = (s.max_y - s.min_y) * pixel_height_m;
            title_ss << " " << width_m << "w " << height_m << "l";
        }

        std::ostringstream sinkhole_notes_ss;
        sinkhole_notes_ss << std::fixed << std::setprecision(1)
                          << "depth: " << s.max_depth << "m\n"
                          << "area: " << area_to_string(s.area, pixel_width_m, pixel_height_m) << "\n"
						  << "elevation: " << s.elevation << "m";

        std::string marker_color = color_to_hex(const_cast<Settings&>(settings).depth_to_color(s.max_depth));
        std::string icon = const_cast<Settings&>(settings).depth_to_gaiagps_color(s.max_depth);

        output["features"].push_back({
            {"type", "Feature"},
            {"geometry", {
                {"type", "Point"},
                {"coordinates", {lon, lat}}
            }},
            {"properties", {
                {"updated_date", now_str},
                {"time_created", now_str},
                {"deleted", false},
                {"title", title_ss.str()},
                {"is_active", true},
                {"icon", icon},
                {"notes", sinkhole_notes_ss.str()},
                {"latitude", lat},
                {"longitude", lon},
                {"marker_type", "pin"},
                {"marker-color", marker_color},
                {"folderId", folder_uuid_hex}
            }}
        });
    }

    std::ofstream file(output_fname);
    file << output.dump(4);
}

void export_sinkholes_geojson(const string& output_fname, const vector<Sinkhole>& sinkholes, const double* geo_transform, const std::string& wkt, Settings& settings)
{
    int max_points = settings.MAX_POINTS_PER_FILE;
    int n = static_cast<int>(sinkholes.size());

    if (max_points > 0 && n > max_points)
    {
        // Split into multiple files: output_0001.geojson, output_0002.geojson, etc.
        size_t dot = output_fname.rfind('.');
        std::string prefix = (dot == std::string::npos) ? output_fname : output_fname.substr(0, dot);
        std::string ext    = (dot == std::string::npos) ? ""            : output_fname.substr(dot);

        int num_files = (n + max_points - 1) / max_points;
        int digits = static_cast<int>(std::ceil(std::log10(num_files + 1)));

        for (int i = 0; i < num_files; i++)
        {
            std::ostringstream chunk_fname;
            chunk_fname << prefix << "_" << std::setw(digits) << std::setfill('0') << i << ext;

            int start = i * max_points;
            int end   = std::min(start + max_points, n);
            std::vector<Sinkhole> chunk(sinkholes.begin() + start, sinkholes.begin() + end);
            write_sinkholes_geojson_chunk(chunk_fname.str(), chunk, geo_transform, wkt, settings);
        }
    }
    else
    {
        write_sinkholes_geojson_chunk(output_fname, sinkholes, geo_transform, wkt, settings);
    }
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
	if (settings.VERBOSE)
	{
    	std::cout << "Finished filling DEM." << std::endl;
	}

	// write modified hillshade if output_hillshade_fname is specified
	if (output_hillshade_fname.has_value())
	{
		// TODO
	}

	// write sinkholes if output_sinkholes_fname is specified
	if (output_sinkholes_fname.has_value())
	{
		string fname = output_sinkholes_fname.value();
		export_sinkholes_geojson(fname, sinkholes, geoTransofrmArgs, wkt, settings);
		 if (settings.VERBOSE)
		{
			std::cout << "Finished writing sinkholes GeoJSON." << std::endl;
		}
	}
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("find_sinkholes", "0.1.0");
    program.add_argument("-i", "--input")
        .help("Input DEM to find sinkholes within.")
        .required();
    program.add_argument("-oh", "--output-hillshade").default_value("")
        .help("Output hillshade raster .tif with sinkholes highlighted. If the flag is used but no filename is provided, the hillshade will be written to the same location as the input DEM with '_hillshade' appended to the filename. If the path provided is a folder, the hillshade will be written to that folder with the same filename as the input DEM, with _hillshade appended to the filename.");
    program.add_argument("-os", "--output-sinkholes").default_value("")
        .help("Output .geojson file with sinkholes. If the flag is used but no filename is provided, the GeoJSON will be written to the same location as the input DEM with .geojson extension. If the path provided is a folder, the GeoJSON will be written to that folder with the same filename as the input DEM, with .geojson extension.");
    
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

	Settings settings = Settings(); // TODO read from settings file if specified

	optional<string> output_hillshade_fname = nullopt;
	if (program.is_used("-oh"))
	{
		string input_fname = program.get<std::string>("-i");
		string output_fname = program.get<std::string>("-oh");
		if (output_fname.empty())
		{
			size_t dot = input_fname.rfind('.');
			string prefix = (dot == std::string::npos) ? input_fname : input_fname.substr(0, dot);
			string ext    = (dot == std::string::npos) ? ""            : input_fname.substr(dot);
			output_fname = prefix + "_hillshade" + ext;
		}
		else
		{
			// check if output_fname is a folder
			struct stat info;
			if (stat(output_fname.c_str(), &info) == 0 && (info.st_mode & S_IFDIR))
			{
				size_t dot = input_fname.rfind('.');
				string prefix = (dot == std::string::npos) ? input_fname : input_fname.substr(0, dot);
				string ext    = (dot == std::string::npos) ? ""            : input_fname.substr(dot);
				output_fname = output_fname + "/" + prefix.substr(prefix.find_last_of("/\\") + 1) + "_hillshade" + ext;
			}

			output_hillshade_fname = output_fname;
		}
	}
	optional<string> output_sinkholes_fname = program.is_used("-os") ? optional<string>(program.get<std::string>("-os")) : std::nullopt;
	if (program.is_used("-os"))
	{
		string input_fname = program.get<std::string>("-i");
		string output_fname = program.get<std::string>("-os");
		if (output_fname.empty())
		{
			size_t dot = input_fname.rfind('.');
			string prefix = (dot == std::string::npos) ? input_fname : input_fname.substr(0, dot);
			output_fname = prefix + ".geojson";
		}
		else
		{
			// check if output_fname is a folder
			struct stat info;
			if (stat(output_fname.c_str(), &info) == 0 && (info.st_mode & S_IFDIR))
			{
				size_t dot = input_fname.rfind('.');
				string prefix = (dot == std::string::npos) ? input_fname : input_fname.substr(0, dot);
				output_fname = output_fname + "/" + prefix.substr(prefix.find_last_of("/\\") + 1) + ".geojson";
			}
		}

		output_sinkholes_fname = output_fname;
	}

	std::cout << "output_hillshade_fname: " << (output_hillshade_fname.has_value() ? output_hillshade_fname.value() : "none") << std::endl;
	std::cout << "output_sinkholes_fname: " << (output_sinkholes_fname.has_value() ? output_sinkholes_fname.value() : "none") << std::endl;

	if (program.is_used("-i"))
	{
    	std::string input_fname = program.get<std::string>("-i");
    	handle_dem(input_fname, output_hillshade_fname, output_sinkholes_fname, settings);
	}
	else
	{
		std::cerr << "Input DEM file must be specified. Exiting." << std::endl;
		std::cerr << program;
		std::exit(1);
	}

    return 0;
}