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

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <curl/curl.h>
#include <pdal/PipelineManager.hpp>

#include "argparse.hpp"
#include "json.hpp"
#include "qgis_integration.h"
#include "settings.h"

namespace fs = std::filesystem;

// Forward declaration from find_sinkholes.cpp
void handle_dem(std::string input_fname,
                std::optional<std::string> output_hillshade_fname,
                std::optional<std::string> output_sinkholes_fname,
                Settings& settings);

// ---------------------------------------------------------------------------
// libcurl helpers
// ---------------------------------------------------------------------------

static size_t curl_write_to_string(void* ptr, size_t size, size_t nmemb, std::string* out)
{
    out->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

static size_t curl_write_to_file(void* ptr, size_t size, size_t nmemb, FILE* out)
{
    return fwrite(ptr, size, nmemb, out);
}

// ---------------------------------------------------------------------------
// download_point_clouds
// ---------------------------------------------------------------------------

std::vector<std::string> download_point_clouds(
    double lower_left_lat,
    double lower_left_lon,
    double upper_right_lat,
    double upper_right_lon,
    const Settings& settings,
    const std::string& usgs_response_output_fname = "")
{
    std::ostringstream url_ss;
    url_ss << "https://tnmaccess.nationalmap.gov/api/v1/products"
           << "?prodFormats=LAS,LAZ"
           << "&datasets=Lidar%20Point%20Cloud%20(LPC)"
           << "&bbox=" << lower_left_lon << "," << lower_left_lat
                       << "," << upper_right_lon << "," << upper_right_lat << ",&";
    std::string url = url_ss.str();

    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("Failed to initialise libcurl");

    std::string response_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
        throw std::runtime_error(std::string("USGS API request failed: ") + curl_easy_strerror(res));

    nlohmann::json product_list = nlohmann::json::parse(response_body);

    if (!usgs_response_output_fname.empty())
    {
        std::ofstream f(usgs_response_output_fname);
        f << product_list.dump(2);
    }

    if (!product_list.contains("items"))
    {
        std::cerr << "Error fetching point cloud list for bbox "
                  << lower_left_lat << "," << lower_left_lon
                  << " to " << upper_right_lat << "," << upper_right_lon
                  << ". Response:\n" << product_list.dump(2) << std::endl;
        return {};
    }

    fs::create_directories(settings.POINT_CLOUDS_DIR);

    auto& items = product_list["items"];
    int num_items = static_cast<int>(items.size());
    std::vector<std::string> downloaded_files;

    for (int i = 0; i < num_items; i++)
    {
        std::string download_url = items[i]["downloadURL"];
        std::string fname = download_url.substr(download_url.rfind('/') + 1);
        std::string output_path = settings.POINT_CLOUDS_DIR + "/" + fname;

        if (settings.VERBOSE)
            std::cout << "\rDownloading point cloud " << (i+1) << "/" << num_items
                      << ": " << fname << std::flush;

        FILE* out_file = fopen(output_path.c_str(), "wb");
        if (!out_file)
        {
            std::cerr << "\nCould not open " << output_path << " for writing" << std::endl;
            continue;
        }

        CURL* dl_curl = curl_easy_init();
        curl_easy_setopt(dl_curl, CURLOPT_URL, download_url.c_str());
        curl_easy_setopt(dl_curl, CURLOPT_WRITEFUNCTION, curl_write_to_file);
        curl_easy_setopt(dl_curl, CURLOPT_WRITEDATA, out_file);
        curl_easy_setopt(dl_curl, CURLOPT_FOLLOWLOCATION, 1L);

        long http_code = 0;
        CURLcode dl_res = curl_easy_perform(dl_curl);
        curl_easy_getinfo(dl_curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(dl_curl);
        fclose(out_file);

        if (dl_res != CURLE_OK || http_code != 200)
        {
            std::cerr << "\nError downloading " << fname
                      << " (HTTP " << http_code << "): "
                      << curl_easy_strerror(dl_res) << std::endl;
            fs::remove(output_path);
            continue;
        }

        downloaded_files.push_back(output_path);
    }

    if (settings.VERBOSE)
        std::cout << std::endl;

    return downloaded_files;
}

// ---------------------------------------------------------------------------
// make_dem
// ---------------------------------------------------------------------------

void make_dem(const std::string& input_laz, const std::string& output_tif, const Settings& settings)
{
    std::ostringstream cls_ss;
    for (size_t i = 0; i < settings.POINT_CLOUD_CLASSIFICATIONS.size(); i++)
    {
        if (i > 0) cls_ss << ", ";
        int c = settings.POINT_CLOUD_CLASSIFICATIONS[i];
        cls_ss << "Classification[" << c << ":" << c << "]";
    }

    nlohmann::json pipeline = nlohmann::json::array({
        {
            {"type", "readers.las"},
            {"filename", input_laz}
        },
        {
            {"type", "filters.range"},
            {"limits", cls_ss.str()}
        },
        {
            {"type", "writers.gdal"},
            {"filename", output_tif},
            {"output_type", "min"},
            {"resolution", settings.RESOLUTION},
            {"nodata", settings.NODATA_VALUE},
            {"data_type", "float32"},
            {"gdaldriver", "GTiff"}
        }
    });

    std::istringstream pipeline_stream(pipeline.dump());
    pdal::PipelineManager manager;
    manager.readPipeline(pipeline_stream);
    manager.execute();
}

// ---------------------------------------------------------------------------
// resolve_output_path
//
// Returns nullopt if the flag was not used.
// If flag_value is empty, derives the filename from dem_path and writes it
// next to the DEM.
// If flag_value ends with the expected extension (e.g. ".tif" or ".geojson"),
// it is treated as an exact file path; its parent directory is created if needed.
// Otherwise flag_value is treated as an output directory, which is created if
// it does not already exist.
// ---------------------------------------------------------------------------

static std::optional<std::string> resolve_output_path(
    bool flag_used,
    const std::string& flag_value,
    const std::string& dem_path,
    const std::string& suffix,
    const std::string& ext)
{
    if (!flag_used)
        return std::nullopt;

    std::string stem = fs::path(dem_path).stem().string();

    if (flag_value.empty())
    {
        std::string parent = fs::path(dem_path).parent_path().string();
        return parent.empty() ? (stem + suffix + ext)
                              : (parent + "/" + stem + suffix + ext);
    }

    bool ends_with_ext = flag_value.size() >= ext.size() &&
                         flag_value.compare(flag_value.size() - ext.size(), ext.size(), ext) == 0;

    if (!ends_with_ext)
    {
        // Treat as output directory: create it and derive filename from DEM stem
        fs::create_directories(flag_value);
        return flag_value + "/" + stem + suffix + ext;
    }

    // Exact file path: create parent directory if it is non-trivial
    fs::path parent = fs::path(flag_value).parent_path();
    if (!parent.empty())
        fs::create_directories(parent);
    return flag_value;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("find_sinkholes", "0.1.0");
    program.add_argument("-ll", "--bbox-lower-left")
        .help("Lower-left corner of bounding box to download LiDAR from: latitude,longitude");
    program.add_argument("-ur", "--bbox-upper-right")
        .help("Upper-right corner of bounding box to download LiDAR from: latitude,longitude");
    program.add_argument("-pc", "--point-clouds")
        .help("Point cloud .laz/.las file(s) to convert to DEMs and process")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-d", "--dem")
        .help("Input DEM .tif file(s) to process directly")
        .nargs(argparse::nargs_pattern::any);
    program.add_argument("-s", "--settings")
        .help("Path to a JSON settings file");
    program.add_argument("-oh", "--output-hillshade").default_value(std::string(""))
        .help("Output hillshade .tif. Omit a value to write next to each input DEM; "
              "supply a directory to write all hillshades there.");
    program.add_argument("-os", "--output-sinkholes").default_value(std::string(""))
        .help("Output sinkholes .geojson. Omit a value to write next to each input DEM; "
              "supply a directory to write all GeoJSON files there.");
    program.add_argument("-q", "--qgis")
        .help("Path to the QGIS project file, overriding the value in the settings file.");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    Settings settings = program.is_used("--settings")
        ? Settings::from_json(program.get<std::string>("--settings"))
        : Settings();

    if (program.is_used("--qgis"))
        settings.QGIS_PROJECT_FILE = program.get<std::string>("--qgis");

    if (!program.is_used("-oh") && !program.is_used("-os"))
    {
        std::cerr << "At least one output option must be specified (-oh or -os)." << std::endl;
        std::cerr << program;
        return 1;
    }

    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;

    std::vector<std::string> dem_files;

    double download_elapsed = 0.0;
    double conversion_elapsed = 0.0;

    // 1. Download from bbox → convert to DEMs
    if (program.is_used("--bbox-lower-left") && program.is_used("--bbox-upper-right"))
    {
        auto parse_latlon = [](const std::string& s) -> std::pair<double, double> {
            size_t comma = s.find(',');
            return {std::stod(s.substr(0, comma)), std::stod(s.substr(comma + 1))};
        };

        auto [ll_lat, ll_lon] = parse_latlon(program.get<std::string>("--bbox-lower-left"));
        auto [ur_lat, ur_lon] = parse_latlon(program.get<std::string>("--bbox-upper-right"));

        auto t0 = Clock::now();
        auto downloaded = download_point_clouds(ll_lat, ll_lon, ur_lat, ur_lon, settings, "response.json");
        download_elapsed = Seconds(Clock::now() - t0).count();

        if (settings.VERBOSE)
            std::cout << "Downloaded " << downloaded.size() << " point cloud files." << std::endl;

        fs::create_directories(settings.DEMS_DIR);
        int n = static_cast<int>(downloaded.size());
        auto tc0 = Clock::now();
        for (int i = 0; i < n; i++)
        {
            const std::string& laz_path = downloaded[i];
            std::string stem = fs::path(laz_path).stem().string();
            std::string dem_path = settings.DEMS_DIR + "/" + stem + ".tif";
            if (settings.VERBOSE)
                std::cout << "\rConverting point cloud " << (i+1) << "/" << n
                          << " to DEM: " << dem_path << std::flush;
            make_dem(laz_path, dem_path, settings);
            dem_files.push_back(dem_path);
        }
        conversion_elapsed += Seconds(Clock::now() - tc0).count();
        if (settings.VERBOSE && n > 0)
            std::cout << std::endl;
    }

    // 2. Convert specified point clouds → DEMs
    if (program.is_used("--point-clouds"))
    {
        fs::create_directories(settings.DEMS_DIR);
        auto pcs = program.get<std::vector<std::string>>("--point-clouds");
        int n = static_cast<int>(pcs.size());
        auto tc0 = Clock::now();
        for (int i = 0; i < n; i++)
        {
            const std::string& laz_path = pcs[i];
            std::string stem = fs::path(laz_path).stem().string();
            std::string dem_path = settings.DEMS_DIR + "/" + stem + ".tif";
            if (settings.VERBOSE)
                std::cout << "\rConverting point cloud " << (i+1) << "/" << n
                          << " to DEM: " << dem_path << std::flush;
            make_dem(laz_path, dem_path, settings);
            dem_files.push_back(dem_path);
        }
        conversion_elapsed += Seconds(Clock::now() - tc0).count();
        if (settings.VERBOSE && n > 0)
            std::cout << std::endl;
    }

    // 3. Use directly specified DEMs
    if (program.is_used("--dem"))
    {
        auto direct = program.get<std::vector<std::string>>("--dem");
        dem_files.insert(dem_files.end(), direct.begin(), direct.end());
    }

    if (dem_files.empty())
    {
        std::cerr << "No DEMs to process. Specify -ll/-ur, -pc, or -d." << std::endl;
        std::cerr << program;
        return 1;
    }

    bool oh_used = program.is_used("-oh");
    bool os_used = program.is_used("-os");
    std::string oh_value = oh_used ? program.get<std::string>("-oh") : "";
    std::string os_value = os_used ? program.get<std::string>("-os") : "";

    // Resolve QGIS launch environment once before the loop.
    bool use_qgis = program.is_used("--qgis");
    QgisLaunchContext qgis_ctx;
    if (use_qgis)
    {
        qgis_ctx = prepare_qgis_launch(argv[0]);
        if (!qgis_ctx.valid())
        {
            std::cerr << "Warning: could not find QGIS Python installation. "
                         "Outputs will not be added to the QGIS project." << std::endl;
            use_qgis = false;
        }
    }

    auto tp0 = Clock::now();
    int total = static_cast<int>(dem_files.size());
    for (int i = 0; i < total; i++)
    {
        const std::string& dem_path = dem_files[i];

        if (settings.VERBOSE)
            std::cout << "Processing DEM " << (i+1) << "/" << total
                      << ": " << dem_path << std::endl;

        auto hillshade_out = resolve_output_path(oh_used, oh_value, dem_path, "_hillshade", ".tif");
        auto sinkholes_out = resolve_output_path(os_used, os_value, dem_path, "", ".geojson");

        handle_dem(dem_path, hillshade_out, sinkholes_out, settings);

        if (use_qgis)
        {
            if (settings.VERBOSE)
                std::cout << "Updating QGIS project: " << settings.QGIS_PROJECT_FILE << std::endl;
            update_qgis_project(qgis_ctx, settings, hillshade_out, sinkholes_out);
        }
    }
    double processing_elapsed = Seconds(Clock::now() - tp0).count();

    if (download_elapsed > 0.0)
        std::cout << std::fixed << std::setprecision(3)
                  << "Download time:                      " << download_elapsed << "s" << std::endl;
    if (conversion_elapsed > 0.0)
        std::cout << std::fixed << std::setprecision(3)
                  << "Point cloud to DEM conversion time: " << conversion_elapsed << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
                  << "DEM processing time:                " << processing_elapsed << "s" << std::endl;

    return 0;
}
