#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

input_folder="$1"
output_folder="$2"

# Make sure the output folder exists
mkdir -p "$output_folder"

# Loop through all .tif files in the input folder
for input_file in "$input_folder"/*.tif; do
    # Skip if no .tif files found
    [ -e "$input_file" ] || continue

    # Extract the base filename (without folder and extension)
    base_name=$(basename "$input_file" .tif)

    # Construct output file paths
    output_tif="${output_folder}/${base_name}.tif"
    output_json="${output_folder}/${base_name}.geojson"

    # Run the command
    ./sinkholes.py -i "$input_file" -otif "$output_tif" -ojson "$output_json"
done