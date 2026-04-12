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

Color nodata_color = {0, 0, 0}; // TODO allow user to change this in settings

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
	float spill_elevation,
    BitArray2d& visited,
    queue<Cell>& depressionQueue,
    queue<Cell>& traceQueue,
    PriorityQueue& priorityQueue,
    Sinkhole& sinkhole,
    std::vector<uint8_t>* hillshade_rgb,
	Settings& settings)
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
			if (neighbor_elevation > spill_elevation) 
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
				if (hillshade_rgb != nullptr)
				{
					Color color;
					if (dem.is_NoData(neighbor_row, neighbor_col))
					{
						color = nodata_color;
					}
					else
					{
						color =  settings.depth_to_color(spill_elevation - neighbor_elevation);
					}
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 0] = color.r;
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 1] = color.g;
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 2] = color.b;
				}

                sinkhole.update(dem, neighbor_row, neighbor_col, spill_elevation);

                visited.set_true(neighbor_row,neighbor_col);
				if (!dem.is_NoData(neighbor_row, neighbor_col))
				{
                	dem.set_value(neighbor_row, neighbor_col, spill_elevation);
				}
                neighbor_node.row = neighbor_row;
                neighbor_node.col = neighbor_col;
                neighbor_node.spill_elevation = spill_elevation;
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

void fill_dem(CDEM& dem, std::vector<Sinkhole>& sinkholes, Settings& settings, std::vector<uint8_t>* hillshade_rgb)
{
    // modified version of FillDEM_Zhou_OnePass

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

			if (visited.is_visited(neighbor_row,neighbor_col))
			{
				continue;
			}

			neighbor_elevation = dem.asFloat(neighbor_row, neighbor_col);
			if (neighbor_elevation <= spill_elevation)
			{
				//depression cell
				if (hillshade_rgb != nullptr)
				{
					Color color;
					if (dem.is_NoData(neighbor_row, neighbor_col))
					{
						color = nodata_color;
					}
					else
					{
						color =  settings.depth_to_color(spill_elevation - neighbor_elevation);
					}
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 0] = color.r;
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 1] = color.g;
					(*hillshade_rgb)[(neighbor_row * width + neighbor_col) * 3 + 2] = color.b;
				}

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

				my_ProcessPit_onepass(dem, spill_elevation, visited, depressionQueue, traceQueue, priorityQueue, sinkhole, hillshade_rgb, settings);

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

    for (const Sinkhole& sinkhole : sinkholes)
    {
        auto [lat, lon] = const_cast<Sinkhole&>(sinkhole).to_wgs84(geo_transform, wkt);

        std::ostringstream title_ss;
        title_ss << std::fixed << std::setprecision(1) << "sinkhole " << sinkhole.max_depth << "d";
        if (pixel_width_m >= 0)
        {
            double width_m  = (sinkhole.max_x - sinkhole.min_x + 1) * pixel_width_m;
            double height_m = (sinkhole.max_y - sinkhole.min_y + 1) * pixel_height_m;
            title_ss << " " << width_m << "w " << height_m << "l";
        }

        std::ostringstream sinkhole_notes_ss;
        sinkhole_notes_ss << std::fixed << std::setprecision(1)
                          << "depth: " << sinkhole.max_depth << "m\n"
                          << "area: " << area_to_string(sinkhole.area, pixel_width_m, pixel_height_m) << "\n"
						  << "elevation: " << sinkhole.elevation << "m";

        std::string marker_color = color_to_hex(const_cast<Settings&>(settings).depth_to_color(sinkhole.max_depth));
        std::string icon = const_cast<Settings&>(settings).depth_to_gaiagps_color(sinkhole.max_depth);

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

// Returns a flat RGB uint8_t buffer, row-major, 3 bytes per pixel (R, G, B).
// Uses Horn's method for slope/aspect and the standard hillshade formula.
// Border pixels replicate their nearest interior neighbor for the gradient computation.
static std::vector<uint8_t> compute_hillshade(const CDEM& dem, const double* geo_transform, const Settings& settings)
{
    int width  = dem.Get_NX();
    int height = dem.Get_NY();

    // Pixel size in CRS units (geotransform[5] is negative for north-up rasters)
    double cell_size_x = std::fabs(geo_transform[1]);
    double cell_size_y = std::fabs(geo_transform[5]);

    double zenith_rad  = (90.0 - settings.HILLSHADE_ALTITUDE) * M_PI / 180.0;
    // Convert geographic azimuth (N=0, clockwise) to math azimuth (E=0, counterclockwise)
    double azimuth_rad = (360.0 - settings.HILLSHADE_AZIMUTH + 90.0) * M_PI / 180.0;

    std::vector<uint8_t> rgb(width * height * 3);

    auto get_z = [&](int row, int col) -> float {
        row = std::max(0, std::min(row, height - 1));
        col = std::max(0, std::min(col, width  - 1));
        float v = dem.asFloat(row, col);
        return dem.is_NoData(row, col) ? 0.0f : v;
    };

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            uint8_t shade = 0;

            if (!dem.is_NoData(row, col))
            {
                // 3x3 neighbourhood (Horn's method)
                //  a b c
                //  d e f   e = center (row, col)
                //  g h i
                float a = get_z(row-1, col-1), b = get_z(row-1, col), c = get_z(row-1, col+1);
                float d = get_z(row,   col-1),                         f = get_z(row,   col+1);
                float g = get_z(row+1, col-1), h = get_z(row+1, col), i = get_z(row+1, col+1);

                double dz_dx = ((c + 2.0*f + i) - (a + 2.0*d + g)) / (8.0 * cell_size_x);
                double dz_dy = ((g + 2.0*h + i) - (a + 2.0*b + c)) / (8.0 * cell_size_y);

                dz_dx *= settings.HILLSHADE_Z_FACTOR;
                dz_dy *= settings.HILLSHADE_Z_FACTOR;

                double slope_rad  = std::atan(std::sqrt(dz_dx*dz_dx + dz_dy*dz_dy));
                double aspect_rad = std::atan2(dz_dy, -dz_dx);

                double val = std::cos(zenith_rad) * std::cos(slope_rad)
                           + std::sin(zenith_rad) * std::sin(slope_rad) * std::cos(azimuth_rad - aspect_rad);

                shade = static_cast<uint8_t>(std::max(0.0, std::min(255.0, 255.0 * (val + 1.0) / 2.0)));
            }

            int idx = (row * width + col) * 3;
            rgb[idx]     = shade;
            rgb[idx + 1] = shade;
            rgb[idx + 2] = shade;
        }
    }

    return rgb;
}

static void write_rgb_tiff(
    const std::string& path,
    const std::vector<uint8_t>& rgb,
    int width,
    int height,
    const double* geo_transform,
    const std::string& wkt)
{
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* ds = driver->Create(path.c_str(), width, height, 3, GDT_Byte, nullptr);
    if (ds == nullptr)
    {
        std::cerr << "Failed to create output hillshade file: " << path << std::endl;
        return;
    }

    ds->SetGeoTransform(const_cast<double*>(geo_transform));
    ds->SetProjection(wkt.c_str());

    // Write all 3 bands in one call using interleaved (R,G,B per pixel) layout
    int band_list[3] = {1, 2, 3};
    CPLErr err = ds->RasterIO(GF_Write, 0, 0, width, height,
        const_cast<uint8_t*>(rgb.data()),
        width, height, GDT_Byte,
        3, band_list,
        3,           // nPixelSpace: bytes between pixels
        width * 3,   // nLineSpace:  bytes between rows
        1,           // nBandSpace:  bytes between bands within one pixel
        nullptr);
    if (err != CE_None)
    {
        std::cerr << "Error writing hillshade to " << path << ": "
                  << CPLGetLastErrorMsg() << std::endl;
        GDALClose(ds);
        return;
    }

    GDALClose(ds);
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
	std::vector<uint8_t> hillshade_rgb;
	if (output_hillshade_fname.has_value())
	{
		hillshade_rgb = compute_hillshade(dem, geoTransofrmArgs, settings);
	}

	// fill DEM and find sinkholes
	std::vector<Sinkhole> sinkholes;
    fill_dem(dem,
		sinkholes,
		settings, 
		output_hillshade_fname.has_value() ? &hillshade_rgb : nullptr);
	if (settings.VERBOSE)
	{
    	std::cout << "Finished filling DEM." << std::endl;
	}

	// write modified hillshade if output_hillshade_fname is specified
	if (output_hillshade_fname.has_value())
	{
		write_rgb_tiff(output_hillshade_fname.value(), hillshade_rgb,
			dem.Get_NX(), dem.Get_NY(), geoTransofrmArgs, wkt);
		if (settings.VERBOSE)
		{
			std::cout << "Finished writing hillshade to " << output_hillshade_fname.value() << "." << std::endl;
		}
	}

	// write sinkholes if output_sinkholes_fname is specified
	if (output_sinkholes_fname.has_value())
	{
		string fname = output_sinkholes_fname.value();
		export_sinkholes_geojson(fname, sinkholes, geoTransofrmArgs, wkt, settings);
		 if (settings.VERBOSE)
		{
			std::cout << "Finished writing sinkholes GeoJSON to " << fname << "." << std::endl;
		}
	}
}

