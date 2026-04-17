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

#include <algorithm>
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

#if defined(_WIN32)
// Given a QGIS installation directory (e.g. C:\Program Files\QGIS 3.38),
// returns the path to python.exe inside apps\Python3*\, or an empty string.
// If multiple Python versions exist, the alphabetically latest is returned
// (which corresponds to the newest version).
static std::string find_python_in_qgis_dir(const fs::path& qgis_dir)
{
    fs::path apps = qgis_dir / "apps";
    std::vector<fs::path> candidates;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(apps, ec))
    {
        std::string name = entry.path().filename().string();
        if (name.size() >= 6 && name.substr(0, 6) == "Python")
        {
            fs::path candidate = entry.path() / "python.exe";
            if (fs::exists(candidate))
                candidates.push_back(candidate);
        }
    }
    if (candidates.empty())
        return "";
    std::sort(candidates.begin(), candidates.end());
    return candidates.back().string();
}

// Scans C:\Program Files for QGIS installations and returns the directory of
// the newest one (determined by alphabetical order of the directory name,
// e.g. "QGIS 3.38" > "QGIS 3.34"), or an empty path if none is found.
static fs::path find_newest_qgis_dir()
{
    std::vector<fs::path> found;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator("C:\\Program Files", ec))
    {
        std::string name = entry.path().filename().string();
        if (name.size() >= 5 && name.substr(0, 5) == "QGIS ")
            found.push_back(entry.path());
    }
    if (found.empty())
        return {};
    std::sort(found.begin(), found.end());
    return found.back();
}

// Given a path anywhere inside a QGIS installation, walks up the directory
// tree to find the root QGIS install directory (the one named "QGIS *"),
// or returns an empty path if not found.
static fs::path find_qgis_root_from_subpath(const fs::path& subpath)
{
    for (fs::path p = subpath; !p.empty() && p != p.parent_path(); p = p.parent_path())
    {
        std::string name = p.filename().string();
        if (name.size() >= 5 && name.substr(0, 5) == "QGIS ")
            return p;
    }
    return {};
}
#endif // _WIN32

// Returns the Python executable that ships with the QGIS installation, or an
// empty string if no QGIS installation is found.
static std::string find_qgis_python_executable()
{
#if defined(__linux__)
    struct Candidate { std::string qgis_python_dir; std::string python_exe; };
    static const std::vector<Candidate> candidates = {
        {"/usr/share/qgis/python",         "/usr/bin/python3"},
        {"/usr/lib/python3/dist-packages", "/usr/bin/python3"},
        {"/usr/lib/qgis/python",           "/usr/bin/python3"},
    };
    for (const auto& c : candidates)
        if (fs::is_directory(c.qgis_python_dir) && fs::exists(c.python_exe))
            return c.python_exe;
    return "";

#elif defined(__APPLE__)
    struct Candidate { std::string qgis_python_dir; std::string python_exe; };
    static const std::vector<Candidate> candidates = {
        {"/Applications/QGIS.app/Contents/Resources/python",
         "/Applications/QGIS.app/Contents/MacOS/bin/python3"},
        {"/Applications/QGIS-LTR.app/Contents/Resources/python",
         "/Applications/QGIS-LTR.app/Contents/MacOS/bin/python3"},
    };
    for (const auto& c : candidates)
        if (fs::is_directory(c.qgis_python_dir) && fs::exists(c.python_exe))
            return c.python_exe;
    return "";

#elif defined(_WIN32)
    fs::path qgis_dir = find_newest_qgis_dir();
    if (qgis_dir.empty())
        return "";
    return find_python_in_qgis_dir(qgis_dir);

#else
    return "";
#endif
}

// Wraps a string in double quotes for use in a shell command.
static std::string q(const std::string& s) { return "\"" + s + "\""; }

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

// Returns the QGIS Python API directory to pass to the script, or an empty
// string to let the script auto-detect it.
static std::string resolve_qgis_python_path(const Settings& settings)
{
    return settings.QGIS_PYTHON_PATH;
}

// Returns the Python executable to use when launching the script.
// Priority: explicit setting > derived from QGIS_PYTHON_PATH > auto-detection.
// Returns an empty string if no suitable Python can be found.
static std::string resolve_python_executable(const Settings& settings)
{
    if (!settings.PYTHON_EXECUTABLE.empty())
        return settings.PYTHON_EXECUTABLE;

    if (!settings.QGIS_PYTHON_PATH.empty())
    {
#if defined(__APPLE__)
        // Typical Mac path: /Applications/QGIS.app/Contents/Resources/python
        // Python lives at:  /Applications/QGIS.app/Contents/MacOS/bin/python3
        fs::path p = fs::path(settings.QGIS_PYTHON_PATH);
        while (!p.empty() && p.filename() != "Contents")
            p = p.parent_path();
        fs::path candidate = p / "MacOS" / "bin" / "python3";
        return fs::exists(candidate) ? candidate.string() : "/usr/bin/python3";

#elif defined(_WIN32)
        // Walk up from the given path to find the QGIS root directory, then
        // find python.exe inside its apps\Python3*\ subdirectory.
        fs::path qgis_root = find_qgis_root_from_subpath(fs::path(settings.QGIS_PYTHON_PATH));
        if (!qgis_root.empty())
            return find_python_in_qgis_dir(qgis_root);
        return "";

#else
        return "/usr/bin/python3";
#endif
    }

    return find_qgis_python_executable();
}

QgisLaunchContext prepare_qgis_launch(const std::string& argv0, const Settings& settings)
{
    QgisLaunchContext ctx;

    ctx.qgis_python_path = resolve_qgis_python_path(settings);
    ctx.python_exe = resolve_python_executable(settings);

    if (ctx.python_exe.empty())
        return ctx; // valid() == false

    ctx.script_path = (fs::path(argv0).parent_path() / "add_to_qgis_project.py").string();

    // Build the environment prefix for the subprocess.
    std::ostringstream env_prefix;

#if defined(__linux__)
    // Do NOT inherit LD_LIBRARY_PATH: a conda environment may have prepended
    // conda lib dirs that contain a libgdal built against newer GEOS symbols
    // than the system has, causing symbol-not-found errors when QGIS's Python
    // tries to load GDAL. Using only the GRASS lib dir gives QGIS what it needs.
    // Similarly, clearing GDAL_DRIVER_PATH prevents conda's GDAL plugin dir
    // from being picked up by QGIS.
    std::string grass_lib = find_grass_lib_dir();
    if (!grass_lib.empty())
        env_prefix << "LD_LIBRARY_PATH=" << grass_lib << " ";
    env_prefix << "GDAL_DRIVER_PATH= ";

#elif defined(_WIN32)
    // QGIS DLLs (qgis_core.dll etc.) live in the QGIS bin\ directory. Without
    // it on PATH, Python can find the .pyd extension modules but cannot load
    // their QGIS dependencies. Derive the QGIS root from the Python exe path
    // and prepend its bin\ directory to PATH for the subprocess.
    fs::path qgis_root = find_qgis_root_from_subpath(fs::path(ctx.python_exe));
    if (!qgis_root.empty())
    {
        std::string qgis_bin = (qgis_root / "bin").string();
        env_prefix << "set \"PATH=" << qgis_bin << ";%PATH%\" && ";
    }
#endif

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

    if (!ctx.qgis_python_path.empty())
        cmd << " --qgis-python-path " << q(ctx.qgis_python_path);

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
