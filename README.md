# Find sinkholes automatically
This script automatically detects sinkholes using LIDAR elevation data from the USGS. Instructions for use are at [caves.science](https://caves.science).

## Docker

A docker image is provided which bundles this tool and all its dependencies. To use it:

```sh
  docker build . --tag sinkholes:latest
  docker run sinkholes --help
```
