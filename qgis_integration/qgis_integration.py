import lxml.etree as etree
import os
from pathlib import Path
import tempfile
import uuid
import zipfile

qgis_encoding = 'utf-8'

def read_qgis_project(project_path) -> etree.ElementTree:
    project_path = os.path.expanduser(project_path)
    if os.path.exists(project_path):
        if project_path.endswith('.qgs'):
            return etree.parse(project_path, encoding=qgis_encoding)
        elif project_path.endswith('.qgz'):
            with zipfile.ZipFile(project_path, 'r') as zip_file:
                unzipped_project_name = os.path.basename(project_path)[:-4] + '.qgs'
                return etree.parse(zip_file.open(unzipped_project_name))
        else:
            raise ValueError(f'Unsupported QGIS project file format: {project_path}. Expected .qgs or .qgz.')
    else:
        raise FileNotFoundError(f'QGIS project file not found: {project_path}.')

def add_sinkholes_layer(qgis_project: etree.ElementTree,
                        geojson_path: str,
                        layer_name: str| None=None,
                        group_name: str='sinkholes') -> None:
    
    if layer_name is None:
        layer_name = f'{Path(geojson_path).stem}_sinkholes'
    layer_id = f'{layer_name}_{uuid.uuid4()}'

    # add to layer-tree-group
    layer_tree_groups = qgis_project.findall(f'.//layer-tree-group[@name="{group_name}"]')
    if len(layer_tree_groups) == 0:
        raise ValueError(f'Group "{group_name}" not found in QGIS project layer tree.')
    elif len(layer_tree_groups) > 1:
        raise ValueError(f'{len(layer_tree_groups)} groups named "{group_name}" found in QGIS project layer tree. There needs to be only 1.')

    layer_tree_group = layer_tree_groups[0]
    layer_tree_layer_attributes = {
        'patch_size': "-1, -1",
        'legend_split_behavior': "0",
        'checked': "Qt::Checked",
        'expanded': "1",
        'legend_expr': "",
        'id': layer_id,
        'providerKey': "ogr",
        'source': geojson_path,
        'name': layer_name
    }
    layer_tree_layer = etree.SubElement(layer_tree_group, 'layer-tree-layer', attrib=layer_tree_layer_attributes)
    custom_properties = etree.SubElement(layer_tree_layer, 'customproperties')
    custom_properties.append(etree.Element('Option'))

    # add to legendgroup
    legend_groups = qgis_project.findall(f'.//legendgroup[@name="{group_name}"]')
    if len(legend_groups) == 0:
        raise ValueError(f'Group "{group_name}" not found in QGIS project legend.')
    elif len(legend_groups) > 1:
        raise ValueError(f'{len(legend_groups)} groups named "{group_name}" found in QGIS project legend. There needs to be only 1.')
    legend_group = legend_groups[0]
    legend_layer_attributes = {
        'checked': 'Qt::Checked',
        'open': 'true',
        'drawingOrder': '-1',
        'showFeatureCount': '0',
        'name': layer_name
    }
    legend_layer = etree.SubElement(legend_group, 'legendlayer', attrib=legend_layer_attributes)
    file_group = etree.SubElement(legend_layer, 'filegroup', attrib={'hidden': 'false', 'open': 'true'})
    etree.SubElement(file_group, 'legendlayerfile', attrib={'layerid': layer_id, 'visible': '1', 'isInOverview': '0'})

    # add to custom-order
    custom_order = qgis_project.find('.//custom-order')
    if custom_order is not None:
        item = etree.SubElement(custom_order, 'item')
        item.text = layer_id
    else:
        print(f'Warning: custom-order element not found in QGIS project. Sinkholes layer "{layer_name}" will not be added to custom order.')
    
    # add to individual-layer-settings
    individual_layer_settings = qgis_project.find('.//individual-layer-settings')
    if individual_layer_settings is not None:
        layer_setting_attrib = {
            'tolerance': '12',
            'minScale': '0',
            'type': '1',
            'id': layer_id,
            'maxScale': '0',
            'units': '1',
            'enabled': '0'
        }
        etree.SubElement(individual_layer_settings, 'layer-setting', attrib=layer_setting_attrib)
    else:
        print(f'Warning: individual-layer-settings element not found in QGIS project. Sinkholes layer "{layer_name}" will not be added to individual layer settings.')
    
    # add to projectlayers
    project_layers = qgis_project.find('.//projectlayers')
    if project_layers is not None:
        sinkholes_template_file = open('qgis_integration/sinkholes_layer_template.xml', 'r')
        sinkholes_maplayer = etree.parse(sinkholes_template_file)
        sinkholes_template_file.close()

        id = sinkholes_maplayer.find('.//id')
        id.text = layer_id

        data_source = sinkholes_maplayer.find('.//datasource')
        data_source.text = geojson_path

        layer_name_tag = sinkholes_maplayer.find('.//layername')
        layer_name_tag.text = layer_name

        project_layers.append(sinkholes_maplayer.getroot())
    else:
        raise ValueError(f'projectlayers element not found in QGIS project. Sinkholes layer "{layer_name}" cannot be added to project layers.')

    # add to layerorder
    layer_order = qgis_project.find('.//layerorder')
    if layer_order is not None:
        etree.SubElement(layer_order, 'layer', attrib={'id': layer_id})
    else:
        raise ValueError(f'layerorder element not found in QGIS project. Sinkholes layer "{layer_name}" cannot be added to layer order.')


def add_hillshades_layer(qgis_project: etree.ElementTree,
                        tif_path: str,
                        layer_name: str | None=None,
                        group_name: str='hillshades') -> None:
    
    if layer_name is None:
        layer_name = f'{Path(tif_path).stem}_hillshade'
    layer_id = f'{layer_name}_{uuid.uuid4()}'

    # add to layer-tree-group
    layer_tree_groups = qgis_project.findall(f'.//layer-tree-group[@name="{group_name}"]')
    if len(layer_tree_groups) == 0:
        raise ValueError(f'Group "{group_name}" not found in QGIS project layer tree.')
    elif len(layer_tree_groups) > 1:
        raise ValueError(f'{len(layer_tree_groups)} groups named "{group_name}" found in QGIS project layer tree. There needs to be only 1.')

    layer_tree_group = layer_tree_groups[0]
    layer_tree_layer_attributes = {
        'patch_size': "-1, -1",
        'legend_split_behavior': "0",
        'checked': "Qt::Checked",
        'expanded': "0",
        'legend_expr': "",
        'id': layer_id,
        'providerKey': "gdal",
        'source': tif_path,
        'name': layer_name
    }
    layer_tree_layer = etree.SubElement(layer_tree_group, 'layer-tree-layer', attrib=layer_tree_layer_attributes)
    custom_properties = etree.SubElement(layer_tree_layer, 'customproperties')
    custom_properties.append(etree.Element('Option'))

    # add to legendgroup
    legend_groups = qgis_project.findall(f'.//legendgroup[@name="{group_name}"]')
    if len(legend_groups) == 0:
        raise ValueError(f'Group "{group_name}" not found in QGIS project legend.')
    elif len(legend_groups) > 1:
        raise ValueError(f'{len(legend_groups)} groups named "{group_name}" found in QGIS project legend. There needs to be only 1.')
    legend_group = legend_groups[0]
    legend_layer_attributes = {
        'checked': 'Qt::Checked',
        'open': 'false',
        'drawingOrder': '-1',
        'showFeatureCount': '0',
        'name': layer_name
    }
    legend_layer = etree.SubElement(legend_group, 'legendlayer', attrib=legend_layer_attributes)
    file_group = etree.SubElement(legend_layer, 'filegroup', attrib={'hidden': 'false', 'open': 'false'})
    etree.SubElement(file_group, 'legendlayerfile', attrib={'layerid': layer_id, 'visible': '1', 'isInOverview': '0'})

    # add to custom-order
    custom_order = qgis_project.find('.//custom-order')
    if custom_order is not None:
        item = etree.SubElement(custom_order, 'item')
        item.text = layer_id
    else:
        print(f'Warning: custom-order element not found in QGIS project. Hillshade layer "{layer_name}" will not be added to custom order.')
    
    # add to projectlayers
    project_layers = qgis_project.find('.//projectlayers')
    if project_layers is not None:
        hillshade_template_file = open('qgis_integration/hillshade_layer_template.xml', 'r')
        hillshade_maplayer = etree.parse(hillshade_template_file)
        hillshade_template_file.close()

        id = hillshade_maplayer.find('.//id')
        id.text = layer_id

        data_source = hillshade_maplayer.find('.//datasource')
        data_source.text = tif_path

        layer_name_tag = hillshade_maplayer.find('.//layername')
        layer_name_tag.text = layer_name

        project_layers.append(hillshade_maplayer.getroot())
    else:
        raise ValueError(f'projectlayers element not found in QGIS project. Hillshade layer "{layer_name}" cannot be added to project layers.')
    
    # add to layerorder
    layer_order = qgis_project.find('.//layerorder')
    if layer_order is not None:
        etree.SubElement(layer_order, 'layer', attrib={'id': layer_id})
    else:
        raise ValueError(f'layerorder element not found in QGIS project. Sinkholes layer "{layer_name}" cannot be added to layer order.')

def save_qgis_project(qgis_project: etree.ElementTree, output_path: str):
    output_path = os.path.expanduser(output_path)
    if output_path.endswith('.qgs'):
        qgis_project.write(output_path)
    elif output_path.endswith('.qgz'):
        if os.path.exists(output_path):
            # python zipfile module doesn't support editing existing zip files,
            # so we have to read the existing zip file and write a new one with the updated project file
            temp_file, temp_file_name = tempfile.mkstemp(suffix='.zip', dir=os.path.dirname(output_path))
            with zipfile.ZipFile(output_path, 'r') as input_zip_file:
                os.close(temp_file)
                unzipped_project_name = os.path.basename(output_path)[:-4] + '.qgs'
                with zipfile.ZipFile(temp_file_name, 'w') as output_zip_file:
                    output_zip_file.comment = input_zip_file.comment
                    for item in input_zip_file.infolist():
                        if item.filename != unzipped_project_name:
                            output_zip_file.writestr(item, input_zip_file.read(item.filename))
                    output_zip_file.writestr(unzipped_project_name, etree.tostring(qgis_project))
            os.remove(output_path)
            os.rename(temp_file_name, output_path)
        else:
            with zipfile.ZipFile(output_path, 'w') as zip_file:
                unzipped_project_name = os.path.basename(output_path)[:-4] + '.qgs'
                zip_file.writestr(unzipped_project_name, etree.tostring(qgis_project))