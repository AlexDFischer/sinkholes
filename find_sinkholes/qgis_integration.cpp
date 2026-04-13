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

#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "qgis_integration.h"
#include "settings.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Returns the most recent /usr/lib/grassXX/lib directory on Linux, or an
// empty string on other platforms or if none is found. QGIS sets
// LD_LIBRARY_PATH to this directory so that its Python bindings can find the
// right shared libraries (e.g. a newer libbrotlidec than the system default).
static std::string find_grass_lib_dir()
{
#if defined(__linux__)
    std::string result;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("/usr/lib", ec))
    {
        std::string name = entry.path().filename().string();
        if (name.size() > 5 && name.substr(0, 5) == "grass")
        {
            fs::path lib = entry.path() / "lib";
            if (fs::is_directory(lib))
                result = lib.string(); // alphabetically later = newer version
        }
    }
    return result;
#else
    return "";
#endif
}

// Returns the Python executable that ships with the QGIS installation, or an
// empty string if no QGIS installation is found.
static std::string find_qgis_python_executable()
{
    struct Candidate
    {
        std::string qgis_python_dir;
        std::string python_exe;
    };

#if defined(__linux__)
    static const std::vector<Candidate> candidates = {
        {"/usr/share/qgis/python",         "/usr/bin/python3"},
        {"/usr/lib/python3/dist-packages", "/usr/bin/python3"},
        {"/usr/lib/qgis/python",           "/usr/bin/python3"},
    };
#elif defined(__APPLE__)
    static const std::vector<Candidate> candidates = {
        {"/Applications/QGIS.app/Contents/Resources/python",
         "/Applications/QGIS.app/Contents/MacOS/bin/python3"},
        {"/Applications/QGIS-LTR.app/Contents/Resources/python",
         "/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3"},
    };
#else
    static const std::vector<Candidate> candidates = {};
#endif

    for (const auto& c : candidates)
        if (fs::is_directory(c.qgis_python_dir) && fs::exists(c.python_exe))
            return c.python_exe;

    return "";
}

// Wraps a string in double quotes for use in a shell command.
static std::string q(const std::string& s) { return "\"" + s + "\""; }

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

QgisLaunchContext prepare_qgis_launch(const std::string& argv0)
{
    QgisLaunchContext ctx;

    ctx.python_exe = find_qgis_python_executable();
    if (ctx.python_exe.empty())
        return ctx;

    ctx.script_path = (fs::path(argv0).parent_path() / "add_to_qgis_project.py").string();

    // Build the environment prefix for the subprocess.
    // We intentionally do NOT inherit LD_LIBRARY_PATH here: the parent process
    // may have been launched with a conda environment active, which prepends
    // conda lib dirs to LD_LIBRARY_PATH. Those dirs contain a conda-built
    // libgdal that was compiled against newer GEOS symbols than the system has,
    // causing symbol-not-found errors when QGIS's Python tries to load GDAL.
    // Using only the GRASS lib dir gives QGIS the libraries it actually needs.
    //
    // Similarly, clearing GDAL_DRIVER_PATH prevents conda's GDAL plugin
    // directory from being picked up by QGIS.
    std::ostringstream env_prefix;
    std::string grass_lib = find_grass_lib_dir();
    if (!grass_lib.empty())
        env_prefix << "LD_LIBRARY_PATH=" << grass_lib << " ";
    env_prefix << "GDAL_DRIVER_PATH= ";
    ctx.ld_library_path_prefix = env_prefix.str();

    return ctx;
}

void update_qgis_project(
    const QgisLaunchContext& ctx,
    const Settings& settings,
    const std::vector<std::string>& hillshade_output_fnames,
    const std::vector<std::string>& sinkholes_output_fnames)
{
    std::ostringstream cmd;
    cmd << ctx.ld_library_path_prefix
        << ctx.python_exe << " " << q(ctx.script_path)
        << " " << q(settings.QGIS_PROJECT_FILE);

    if (!hillshade_output_fnames.empty())
    {
        cmd << " --hillshades";
        for (const std::string& fname : hillshade_output_fnames)
            cmd << " " << q(fname);

        cmd << " --hillshades-group " << q(settings.HILLSHADE_QGIS_GROUP_NAME);
    }

    if (!sinkholes_output_fnames.empty())
    {
        cmd << " --sinkholes";
        for (const std::string& fname : sinkholes_output_fnames)
            cmd << " " << q(fname);
        cmd << " --sinkholes-group " << q(settings.SINKHOLES_QGIS_GROUP_NAME);
        cmd << " --sinkholes-style " << q(settings.SINKHOLES_QGIS_STYLE_FILE);
    }

    int ret = std::system(cmd.str().c_str());
    if (ret != 0)
        std::cerr << "Warning: QGIS project update failed (exit code " << ret << ")" << std::endl;
}
