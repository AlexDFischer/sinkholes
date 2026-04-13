# Copyright (C) 2026 Alex Fischer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import glob
import os
import sys


# Candidate locations for the QGIS Python API, ordered by likelihood.
_QGIS_PYTHON_CANDIDATES = {
    'linux': [
        '/usr/share/qgis/python',
        '/usr/lib/python3/dist-packages',
        '/usr/lib/qgis/python',
    ],
    'darwin': [
        '/Applications/QGIS.app/Contents/Resources/python',
        '/Applications/QGIS-LTR.app/Contents/Resources/python',
    ],
    'win32': [
        # QGIS installs under "QGIS <version>" — glob for any version
        r'C:/Program Files/QGIS */apps/qgis/python',
    ],
}


def _find_qgis_python():
    """Return the first existing QGIS Python path for this platform, or None."""
    candidates = _QGIS_PYTHON_CANDIDATES.get(sys.platform, [])
    for pattern in candidates:
        # Use glob so Windows version-wildcard patterns work; also handles plain paths
        matches = glob.glob(pattern)
        for path in matches:
            if os.path.isdir(path) and 'qgis' in os.listdir(path):
                return path
    return None


def _init_qgis():
    """Add the QGIS Python API to sys.path and return an initialised QgsApplication."""
    qgis_python_path = _find_qgis_python()
    if qgis_python_path is None:
        sys.exit(
            'Error: could not find the QGIS Python API on this system.\n'
            'Install QGIS (https://qgis.org) and try again.\n'
            'If QGIS is installed in a non-standard location, add its Python '
            'directory to PYTHONPATH before running this script.'
        )

    if qgis_python_path not in sys.path:
        sys.path.insert(0, qgis_python_path)

    try:
        from qgis.core import QgsApplication
    except ImportError as e:
        sys.exit(
            f'Error: found QGIS Python directory at {qgis_python_path!r} '
            f'but could not import qgis.core: {e}\n'
            'Your QGIS installation may be incomplete or incompatible with '
            'this Python version.'
        )

    app = QgsApplication([], False)
    app.initQgis()
    return app


def get_or_create_group(root, name):
    group = root.findGroup(name)
    if group is None:
        group = root.addGroup(name)
    return group


def main():
    parser = argparse.ArgumentParser(
        description='Add sinkhole finder outputs to a QGIS project file.')
    parser.add_argument('project',
        help='Path to the QGIS project file (.qgz). Created if it does not exist.')
    parser.add_argument('--sinkholes', nargs='+', metavar='FILE',
        help='Sinkholes .geojson file(s) to add as vector layers.')
    parser.add_argument('--sinkholes-style',
        help='QML style file to apply to each sinkholes layer.')
    parser.add_argument('--sinkholes-group',
        help='Layer group to add the sinkholes layers into.')
    parser.add_argument('--hillshades', nargs='+', metavar='FILE',
        help='Hillshade .tif file(s) to add as raster layers.')
    parser.add_argument('--hillshades-group',
        help='Layer group to add the hillshade layers into.')
    args = parser.parse_args()

    if not args.sinkholes and not args.hillshade:
        parser.error('At least one of --sinkholes or --hillshade must be specified.')

    app = _init_qgis()

    from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer

    project = QgsProject.instance()
    if os.path.exists(args.project):
        project.read(args.project)
    else:
        project.setFileName(args.project)

    root = project.layerTreeRoot()

    # ---------------------------------------------------------------------------
    # Sinkholes vector layers
    # ---------------------------------------------------------------------------
    if args.sinkholes:
        group = get_or_create_group(root, args.sinkholes_group) \
            if args.sinkholes_group else root
        for path in args.sinkholes:
            name = os.path.splitext(os.path.basename(path))[0]
            layer = QgsVectorLayer(path, name, 'ogr')
            if not layer.isValid():
                print(f'Error: could not load sinkholes layer from {path}',
                      file=sys.stderr)
            else:
                if args.sinkholes_style:
                    layer.loadNamedStyle(args.sinkholes_style)
                project.addMapLayer(layer, False)
                group.addLayer(layer)

    # ---------------------------------------------------------------------------
    # Hillshade raster layers
    # ---------------------------------------------------------------------------
    if args.hillshades:
        group = get_or_create_group(root, args.hillshades_group) \
            if args.hillshades_group else root
        for path in args.hillshades:
            name = os.path.splitext(os.path.basename(path))[0]
            layer = QgsRasterLayer(path, name)
            if not layer.isValid():
                print(f'Error: could not load hillshade layer from {path}',
                      file=sys.stderr)
            else:
                project.addMapLayer(layer, False)
                group.addLayer(layer)

    project.write()
    app.exitQgis()


if __name__ == '__main__':
    main()
