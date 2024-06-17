#!/bin/bash

input_folder=inputDEMs/kangarooMountain

output_folder=output/kangarooMountain

for filename in "$input_folder"/*.tif; do
    output_geotiff_filename=$output_folder/$(basename "$filename")
    output_geojson_filename=$output_folder/$(basename "$filename" .tif).geojson
    ./sinkholes.py -i $filename -otif $output_geotiff_filename -ojson $output_geojson_filename
done