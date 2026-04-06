#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include "argparse.hpp"
#include "dem.h"
#include "Node.h"
#include "utils.h"
#include <time.h>
#include <list>
#include <unordered_map>


using namespace std;
using std::cout;
using std::endl;
using std::string;
using std::getline;
using std::fstream;
using std::ifstream;
using std::priority_queue;
using std::binary_function;

int main(int argc, char** argv)
{
    CDEM dem;
    double geoTransformArgs[6] = {656000, 1, 0, 3.928e+06, 0, -1};
    dem.SetHeight(7);
    dem.SetWidth(7);
    dem.Allocate();
    dem.initialElementsNodata();

    // outer edge set to 3
    for (int row = 0; row < 7; row++)
    {
        dem.set_value(row, 0, 3);
        dem.set_value(row, 6, 3);
    }
    for (int col = 0; col < 7; col++)
    {
        dem.set_value(0, col, 3);
        dem.set_value(6, col, 3);
    }
    // second outer-most edge set to 2
    for (int row = 1; row < 6; row++)
    {
        dem.set_value(row, 1, 2);
        dem.set_value(row, 5, 2);
    }
    for (int col = 1; col < 6; col++)
    {
        dem.set_value(1, col, 2);
        dem.set_value(5, col, 2);   
    }
    // third outer-most edge set to 1
    for (int row = 2; row < 5; row++)
    {
        dem.set_value(row, 2, 1);
        dem.set_value(row, 4, 1);
    }
    for (int col = 2; col < 5; col++)
    {
        dem.set_value(2, col, 1);
        dem.set_value(4, col, 1);
    }
    // inner cell set to nodata
    dem.set_value(3, 3, NO_DATA_VALUE);

    CreateGeoTIFF("bug_mwe_center_odata.tif", dem.Get_NY(), dem.Get_NX(), 
        (void *)dem.getDEMdata(),GDALDataType::GDT_Float32, geoTransformArgs,
        nullptr, nullptr, nullptr, nullptr, NO_DATA_VALUE);
}