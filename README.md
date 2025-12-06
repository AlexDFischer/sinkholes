# Find sinkholes automatically
This script automatically detects sinkholes using LIDAR elevation data from the USGS. Instructions for use are at [caves.science](https://caves.science).

## Docker

A docker image is provided which bundles this tool and all its dependencies. To use it:

```sh
  # Builds the docker image locally
  docker build . --tag sinkholes:latest
  # Runs the docker image. Mounts the input/output folders to keep assets out of the image
  # and cleans itself up once it has run.
  docker run --rm \
    -v "$(pwd)/inputDEMs:/sinkholes/inputDEMs:ro" \
    -v "$(pwd)/output:/sinkholes/output" \
    sinkholes:latest \
    -c config.jsonc \
    -i inputDEMs/<your_file>.tif \
    -otif output/<your_file>.out.tif \
    -ojson output/<your_file>.geojson
```
