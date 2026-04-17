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

#ifndef QGIS_INTEGRATION_H
#define QGIS_INTEGRATION_H

#include <optional>
#include <string>

#include "settings.h"

// Holds the pre-computed state needed to launch add_to_qgis_project.py.
struct QgisLaunchContext
{
    std::string python_exe;
    std::string script_path;
    std::string ld_library_path_prefix; // empty if not needed
    std::string qgis_python_path;       // empty = let script auto-detect

    bool valid() const { return !python_exe.empty(); }
};

// Searches for a QGIS installation, the matching Python executable, and any
// required LD_LIBRARY_PATH prefix. argv0 is used to locate
// add_to_qgis_project.py relative to the binary.
// Returns a context where valid() == false if no QGIS installation is found.
QgisLaunchContext prepare_qgis_launch(const std::string& argv0, const Settings& settings);

// Runs add_to_qgis_project.py to add the given outputs to the QGIS project
// specified in settings.
void update_qgis_project(
    const QgisLaunchContext& ctx,
    const Settings& settings,
    const std::vector<std::string>& hillshade_output_fnames,
    const std::vector<std::string>& sinkholes_output_fnames);

#endif
