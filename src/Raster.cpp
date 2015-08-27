/******************************************************************************
* Copyright (c) 2015, Bradley J Chambers, brad.chambers@gmail.com
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#pragma once

#include <pdal/Raster.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include "gdal_priv.h" // For File I/O
#include "ogr_spatialref.h"  //For Geographic Information/Transformations

#include <pdal/Dimension.hpp>
#include <pdal/PointView.hpp>

namespace pdal
{

namespace Utils
{

const double c_pi = 3.14159265358979323846; /*!< PI value */

typedef std::chrono::high_resolution_clock HiResClock;
typedef std::chrono::time_point<HiResClock> HiResTimePt;

double GetTime(HiResTimePt start, HiResTimePt stop)
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
}

void CreateSurface(Eigen::MatrixXd& dem, const PointViewPtr data,
                   double spacing_x, double spacing_y, uint32_t rows,
                   uint32_t cols, float background, BOX2D extent)
{
    double yMax = extent.miny + rows * spacing_y;

    Eigen::MatrixXi occ = Eigen::MatrixXi::Zero(rows, cols);

    dem.setConstant(background);
    for (PointId idx = 0; idx < data->size(); ++idx)
    {
        using namespace Dimension::Id;
        double x = data->getFieldAs<double>(X, idx);
        double y = data->getFieldAs<double>(Y, idx);
        double z = data->getFieldAs<double>(Z, idx);

        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int col = clamp(static_cast<int>(floor((x - extent.minx) / spacing_x)), 0, cols-1);
        int row = clamp(static_cast<int>(floor((yMax - y) / spacing_y)), 0, rows-1);

        double val = dem(row, col);

        occ(row, col) = 1;

        if (val == background)
        {
            dem(row, col) = z;
        }
        else
        {
            if (z > val)
                dem(row, col) = z;
        }
    }

    int no = 0;
    for (int c = 0; c < cols; ++c)
    {
        for (int r = 0; r < rows; ++r)
        {
            if (occ(r, c)==1) no++;
        }
    }

    int np = data->size();
    int nc = rows * cols;
    double cs = spacing_x * spacing_y;
    double area = no * cs;
    double rd = np / area;
    double rs = 1.0 / std::sqrt(rd);
    double ed = np / ((extent.maxx-extent.minx)*(extent.maxy-extent.miny));
    double es = 1.0 / std::sqrt(ed);

    std::cerr << np << " points occupy " << no << " of " << nc << " cells\n";
    std::cerr << "estimated point density is: " << ed << " pts / m^2" << std::endl;
    std::cerr << "estimated point spacing is: " << es << " m" << std::endl;
    std::cerr << "  refined point density is: " << rd << " pts / m^2" << std::endl;
    std::cerr << "  refined point spacing is: " << rs << " m" << std::endl;
}

void Raster::calculateGridSizes()
{
    BOX2D& extent = getBounds();

    m_cols = (int)(ceil((extent.maxx - extent.minx)/m_spacing_x)) + 1;
    m_rows = (int)(ceil((extent.maxy - extent.miny)/m_spacing_y)) + 1;
}

Raster::Raster(const PointViewPtr data, double spacing_x, double spacing_y,
               SpatialReference inSRS, float c_background, LogPtr log)
{
    setBounds(BOX2D());

    m_background = c_background;
    m_spacing_x = spacing_x;
    m_spacing_y = spacing_y;
    m_inSRS = inSRS;
    m_log = log;

    GDALAllRegister();

    data->calculateBounds(m_bounds);

    // calculate grid based off bounds and post spacing
    calculateGridSizes();
    m_log->get(LogLevel::Debug2) << "X columns: " << m_cols << std::endl;
    m_log->get(LogLevel::Debug2) << "Y rows: " << m_rows << std::endl;

    m_log->floatPrecision(6);
    m_log->get(LogLevel::Debug2) << "X post spacing: " << m_spacing_x << std::endl;
    m_log->get(LogLevel::Debug2) << "Y post spacing: " << m_spacing_y << std::endl;
    m_log->clearFloat();

    BOX2D& extent = getBounds();

    // need to create the max DEM
    m_dem = Eigen::MatrixXd(m_rows, m_cols);

    m_log->get(LogLevel::Debug) << "Creating... ";
    auto start = std::chrono::high_resolution_clock::now();
    CreateSurface(m_dem, data, m_spacing_x, m_spacing_y, m_rows,
                  m_cols, m_background, extent);
    auto stop = std::chrono::high_resolution_clock::now();
    m_log->get(LogLevel::Debug) << "done in " << GetTime(start, stop) << " ms\n";

    auto CleanRasterScanLine = [](Eigen::MatrixXd& data,
                                  int mDim, int row,
                                  double background, bool* prevSetCols,
                                  bool* curSetCols)
    {

        auto InterpolateRasterPixelScanLine = [](const Eigen::MatrixXd& data,
                                              int mDim, int row, int col,
                                              double background,
                                              bool* prevSetCols)
        {
            int rowMinus, rowPlus, colMinus, colPlus;
            float tInterpValue;
            bool tPrevInterp;

            rowMinus = row - 1;
            rowPlus = row + 1;
            colMinus = col - 1;
            colPlus = col + 1;

            //North
            tInterpValue = data(rowMinus, col);
            tPrevInterp = prevSetCols[col];
            if (tInterpValue != background && tPrevInterp != true)
                return tInterpValue;

            //South
            tInterpValue = data(rowPlus, col);
            if (tInterpValue != background)
                return tInterpValue;

            //East
            tInterpValue = data(row, colPlus);
            if (tInterpValue != background)
                return tInterpValue;

            //West
            tInterpValue = data(row, colMinus);
            if (tInterpValue != background)
                return tInterpValue;

            //NorthWest
            tInterpValue = data(rowMinus, colMinus);
            tPrevInterp = prevSetCols[colMinus];
            if (tInterpValue != background && tPrevInterp != true)
                return tInterpValue;

            //NorthWest
            tInterpValue = data(rowMinus, colPlus);
            tPrevInterp = prevSetCols[colPlus];
            if (tInterpValue != background && tPrevInterp != true)
                return tInterpValue;

            //SouthWest
            tInterpValue = data(rowPlus, colMinus);
            if (tInterpValue != background)
                return tInterpValue;

            //SouthEast
            tInterpValue = data(rowPlus, colPlus);
            if (tInterpValue != background)
                return tInterpValue;

            //ABELL - Returning something.
            return 0.0f;
        };

        float tInterpValue;
        float tValue;

        Eigen::MatrixXd interp = data.row(row);

        for (int col = 1; col < mDim-1; ++col)
        {
            tValue = interp(col);

            if (tValue == background)
            {
                tInterpValue =
                    InterpolateRasterPixelScanLine(data, mDim, row, col,
                                                   background, prevSetCols);

                if (tInterpValue != background)
                {
                    curSetCols[col] = true;
                    interp(col) = tInterpValue;
                }
            }
        }

        data.row(row) = interp;
    };

    bool* prevSetCols = new bool[m_cols];
    bool* curSetCols = new bool[m_cols];

    m_log->get(LogLevel::Debug) << "Cleaning... ";
    start = std::chrono::high_resolution_clock::now();
    for (uint32_t row = 1; row < m_rows - 1; ++row)
    {
        CleanRasterScanLine(m_dem, m_cols, row,
                            m_background, prevSetCols, curSetCols);
        memcpy(prevSetCols, curSetCols, m_cols);
        memset(curSetCols, 0, m_cols);
    }
    stop = std::chrono::high_resolution_clock::now();
    m_log->get(LogLevel::Debug) << "done in " << GetTime(start, stop) << " ms\n";

    delete[] prevSetCols;
    delete[] curSetCols;
}

double Raster::GetNeighbor(const Eigen::MatrixXd &data, int row, int col,
                           Direction d)
{
    double val;
    switch (d)
    {
        case NORTH:
            val = data(row-1, col);
            break;
        case SOUTH:
            val = data(row+1, col);
            break;
        case EAST:
            val = data(row, col+1);
            break;
        case WEST:
            val = data(row, col-1);
            break;
        case NORTHEAST:
            val = data(row-1, col+1);
            break;
        case NORTHWEST:
            val = data(row-1, col-1);
            break;
        case SOUTHEAST:
            val = data(row+1, col+1);
            break;
        case SOUTHWEST:
            val = data(row+1, col-1);
            break;
        default:
            val = data(row, col);
            break;
    }
    return val;
}

double Raster::determineSlopeFD(const Eigen::MatrixXd &data, int row, int col,
                                double postSpacing, double valueToIgnore)
{
    double tSlopeVal = valueToIgnore;
    double tSlopeValDegree = valueToIgnore;

    double mean = 0.0;
    unsigned int nvals = 0;

    double val = static_cast<double>(data(row, col));
    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);

    auto accumulate = [&nvals, &mean, valueToIgnore](double val)
    {
        if (val != valueToIgnore)
        {
            mean += val;
            nvals++;
        }
    };

    accumulate(val);
    accumulate(north);
    accumulate(south);
    accumulate(east);
    accumulate(west);

    mean /= nvals;

    if (north == valueToIgnore) north = mean;
    if (south == valueToIgnore) south = mean;
    if (east == valueToIgnore) east = mean;
    if (west == valueToIgnore) west = mean;

    double zX = (east - west) / (2 * postSpacing);
    double zY = (north - south) / (2 * postSpacing);
    double p = (zX * zX) + (zY * zY);

    tSlopeVal = std::sqrt(p);

    if (tSlopeVal != valueToIgnore)
        tSlopeValDegree = atan(tSlopeVal) * (180.0f / c_pi);

    return tSlopeValDegree;
}

double Raster::determineSlopeD8(const Eigen::MatrixXd &data, int row, int col,
                                double postSpacing, double valueToIgnore)
{
    double tPhi1 = 1.0f;
    double tPhi2 = sqrt(2.0f);
    double tSlopeVal = valueToIgnore;
    double tSlopeValDegree = valueToIgnore;

    double val = static_cast<double>(data(row, col));

    if (val == valueToIgnore)
        return val;

    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);
    double northeast = GetNeighbor(data, row, col, NORTHEAST);
    double northwest = GetNeighbor(data, row, col, NORTHWEST);
    double southeast = GetNeighbor(data, row, col, SOUTHEAST);
    double southwest = GetNeighbor(data, row, col, SOUTHWEST);

    auto checkVal =
        [val, &tSlopeVal, valueToIgnore, postSpacing](double neighbor, double phi)
    {
        if (neighbor != valueToIgnore)
        {
            neighbor = (val - neighbor) / (postSpacing * phi);
            if (std::fabs(neighbor) > std::fabs(tSlopeVal) || tSlopeVal == valueToIgnore)
                tSlopeVal = neighbor;
        }
    };

    checkVal(north, tPhi1);
    checkVal(south, tPhi1);
    checkVal(east, tPhi1);
    checkVal(west, tPhi1);
    checkVal(northeast, tPhi2);
    checkVal(northwest, tPhi2);
    checkVal(southeast, tPhi2);
    checkVal(southwest, tPhi2);

    if (tSlopeVal != valueToIgnore)
        tSlopeValDegree = atan(tSlopeVal) * (180.0f / c_pi);

    return tSlopeValDegree;
}


double Raster::determineAspectFD(const Eigen::MatrixXd& data, int row, int col,
                                 double postSpacing, double valueToIgnore)
{
    double mean = 0.0;
    unsigned int nvals = 0;

    double val = static_cast<double>(data(row, col));
    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);

    auto accumulate = [&nvals, &mean, valueToIgnore](double val)
    {
        if (val != valueToIgnore)
        {
            mean += val;
            nvals++;
        }
    };

    accumulate(val);
    accumulate(north);
    accumulate(south);
    accumulate(east);
    accumulate(west);

    mean /= nvals;

    if (north == valueToIgnore) north = mean;
    if (south == valueToIgnore) south = mean;
    if (east == valueToIgnore) east = mean;
    if (west == valueToIgnore) west = mean;

    double zX = (east - west) / (2 * postSpacing);
    double zY = (north - south) / (2 * postSpacing);
    double p = (zX * zX) + (zY * zY);

    return 180.0 - std::atan(zY/zX) + 90.0 * (zX / std::fabs(zX));
}

double Raster::determineAspectD8(const Eigen::MatrixXd& data, int row,
                                 int col, double postSpacing)
{
    double tPhi1 = 1.0f;
    double tPhi2 = sqrt(2.0f);
    double tH = postSpacing;
    double tVal, tN, tS, tE, tW, tNW, tNE, tSW, tSE, nextTVal;
    double tSlopeVal = std::numeric_limits<double>::max(), tSlopeValDegree;
    int tNextY, tNextX;
    unsigned int j = 0;

    tVal = data(row, col);
    if (tVal == std::numeric_limits<double>::max())
        return tVal;

    //North
    nextTVal = GetNeighbor(data, row, col, NORTH);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tN = (tVal - nextTVal) / (tH * tPhi1);
        if (tN > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tN;
            j = 8;
        }
    }
    //South
    nextTVal = GetNeighbor(data, row, col, SOUTH);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tS = (tVal - nextTVal) / (tH * tPhi1);
        if (tS > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tS;
            j = 4;
        }
    }
    //East
    nextTVal = GetNeighbor(data, row, col, EAST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tE = (tVal - nextTVal) / (tH * tPhi1);
        if (tE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tE;
            j = 2;
        }
    }
    //West
    nextTVal = GetNeighbor(data, row, col, WEST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tW = (tVal - nextTVal) / (tH * tPhi1);
        if (tW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tW;
            j = 6;
        }
    }
    //NorthEast
    nextTVal = GetNeighbor(data, row, col, NORTHEAST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tNE = (tVal - nextTVal) / (tH * tPhi2);
        if (tNE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tNE;
            j = 1;
        }
    }
    //NorthWest
    nextTVal = GetNeighbor(data, row, col, NORTHWEST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tNW = (tVal - nextTVal) / (tH * tPhi2);
        if (tNW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tNW;
            j = 7;
        }
    }
    //SouthEast
    nextTVal = GetNeighbor(data, row, col, SOUTHEAST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tSE = (tVal - nextTVal) / (tH * tPhi2);
        if (tSE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tSE;
            j = 3;
        }
    }
    //SouthWest
    nextTVal = GetNeighbor(data, row, col, SOUTHWEST);
    if (nextTVal < std::numeric_limits<double>::max())
    {
        tSW = (tVal - nextTVal) / (tH * tPhi2);
        if (tSW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        {
            tSlopeVal = tSW;
            j = 5;
        }
    }

    //tSlopeValDegree = 45 * j;
    tSlopeValDegree = std::pow(2.0,j-1);

    return tSlopeValDegree;
}

int Raster::determineCatchmentAreaD8(const Eigen::MatrixXd& data,
                                     Eigen::MatrixXd& area, int row, int col,
                                     double postSpacing)
{
    if (area(row, col) > 0)
    {
        return area(row, col);
    }
    else
    {
        area(row, col) = 1;

        for (int i = 1; i < 9; ++i)
        {
            int j, k;
            switch (i)
            {
                case 1:
                    j = row - 1;
                    k = col + 1;
                    break;

                case 2:
                    j = row;
                    k = col + 1;
                    break;

                case 3:
                    j = row + 1;
                    k = col + 1;
                    break;

                case 4:
                    j = row + 1;
                    k = col;
                    break;

                case 5:
                    j = row + 1;
                    k = col - 1;
                    break;

                case 6:
                    j = row;
                    k = col - 1;
                    break;

                case 7:
                    j = row - 1;
                    k = col - 1;
                    break;

                case 8:
                    j = row - 1;
                    k = col;
                    break;
            }

            if (area(j, k) > 0)
                area(row, col) += determineCatchmentAreaD8(data, area, j, k,
                                  postSpacing);

            // not quite complete here...
        }

        //double tPhi1 = 1.0f;
        //double tPhi2 = sqrt(2.0f);
        //double tH = postSpacing;
        //double tVal, tN, tS, tE, tW, tNW, tNE, tSW, tSE, nextTVal;
        //double tSlopeVal = std::numeric_limits<double>::max(), tSlopeValDegree;
        //int tNextY, tNextX;
        //unsigned int j = 0;

        //tVal = (*data)(row, col);
        //if (tVal == std::numeric_limits<double>::max())
        //  return tVal;

        ////North
        //tNextY = row - 1;
        //tNextX = col;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tN = (tVal - nextTVal) / (tH * tPhi1);
        //  if (tN > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tN;
        //    j = 8;
        //  }
        //}
        ////South
        //tNextY = row + 1;
        //tNextX = col;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tS = (tVal - nextTVal) / (tH * tPhi1);
        //  if (tS > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tS;
        //    j = 4;
        //  }
        //}
        ////East
        //tNextY = row;
        //tNextX = col + 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tE = (tVal - nextTVal) / (tH * tPhi1);
        //  if (tE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tE;
        //    j = 2;
        //  }
        //}
        ////West
        //tNextY = row;
        //tNextX = col - 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tW = (tVal - nextTVal) / (tH * tPhi1);
        //  if (tW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tW;
        //    j = 6;
        //  }
        //}
        ////NorthEast
        //tNextY = row - 1;
        //tNextX = col + 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tNE = (tVal - nextTVal) / (tH * tPhi2);
        //  if (tNE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tNE;
        //    j = 1;
        //  }
        //}
        ////NorthWest
        //tNextY = row - 1;
        //tNextX = col - 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tNW = (tVal - nextTVal) / (tH * tPhi2);
        //  if (tNW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tNW;
        //    j = 7;
        //  }
        //}
        ////SouthEast
        //tNextY = row + 1;
        //tNextX = col + 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tSE = (tVal - nextTVal) / (tH * tPhi2);
        //  if (tSE > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tSE;
        //    j = 3;
        //  }
        //}
        ////SouthWest
        //tNextY = row + 1;
        //tNextX = col - 1;
        //nextTVal = (*data)(tNextY, tNextX);
        //if (nextTVal < std::numeric_limits<double>::max())
        //{
        //  tSW = (tVal - nextTVal) / (tH * tPhi2);
        //  if (tSW > tSlopeVal || tSlopeVal == std::numeric_limits<double>::max())
        //  {
        //    tSlopeVal = tSW;
        //    j = 5;
        //  }
        //}

        //switch (j)
        //{
        //case 1:
        //  tNextY = row - 1;
        //  tNextX = col + 1;
        //  break;

        //case 2:
        //  tNextY = row;
        //  tNextX = col + 1;
        //  break;

        //case 3:
        //  tNextY = row + 1;
        //  tNextX = col + 1;
        //  break;

        //case 4:
        //  tNextY = row + 1;
        //  tNextX = col;
        //  break;

        //case 5:
        //  tNextY = row + 1;
        //  tNextX = col - 1;
        //  break;

        //case 6:
        //  tNextY = row;
        //  tNextX = col - 1;
        //  break;

        //case 7:
        //  tNextY = row - 1;
        //  tNextX = col - 1;
        //  break;

        //case 8:
        //  tNextY = row - 1;
        //  tNextX = col;
        //  break;
        //}
        //(*area)(row, col) = determineCatchmentAreaD8(data, area, tNextY, tNextX, postSpacing);
    }
    //ABELL - Returning something.
    return 0;
}

double Raster::determineHillshade(const Eigen::MatrixXd& data, int row,
                                  int col, double zenithRad, double azimuthRad,
                                  double postSpacing)
{
    //ABELL - tEVar not currently used.
    //double tAVar, tBVar, tCVar, tDVar, tEVar, tFVar, tGVar, tHVar, tIVar;
    double tAVar, tBVar, tCVar, tDVar, tFVar, tGVar, tHVar, tIVar;
    double tDZDX, tDZDY, tSlopeRad, tAspectRad = 0.0;
    double tHillShade;

    tAVar = GetNeighbor(data, row, col, NORTHWEST);
    tBVar = GetNeighbor(data, row, col, NORTH);
    tCVar = GetNeighbor(data, row, col, NORTHWEST);
    tDVar = GetNeighbor(data, row, col, WEST);
    //tEVar = (double)(*data)(row, col);
    tFVar = GetNeighbor(data, row, col, EAST);
    tGVar = GetNeighbor(data, row, col, SOUTHWEST);
    tHVar = GetNeighbor(data, row, col, SOUTH);
    tIVar = GetNeighbor(data, row, col, SOUTHEAST);

    tDZDX = ((tCVar + 2 * tFVar + tIVar) - (tAVar + 2 * tDVar + tGVar)) /
            (8 * postSpacing);
    tDZDY = ((tGVar + 2* tHVar + tIVar) - (tAVar + 2 * tBVar + tCVar))  /
            (8 * postSpacing);
    tSlopeRad = atan(sqrt(pow(tDZDX, 2) + pow(tDZDY, 2)));

    if (tDZDX == 0)
    {
        if (tDZDY > 0)
        {
            tAspectRad = c_pi / 2;
        }
        else if (tDZDY < 0)
        {
            tAspectRad = (2 * c_pi) - (c_pi / 2);
        }
        else
        {
            //ABELL - This looks wrong.  At least needs a comment.
            // tAspectRad = tAspectRad;
            ;
        }
    }
    else
    {
        tAspectRad = atan2(tDZDY, -1 * tDZDX);
        if (tAspectRad < 0)
        {
            tAspectRad = 2 * c_pi + tAspectRad;
        }
    }

    tHillShade = (((cos(zenithRad) * cos(tSlopeRad)) + (sin(zenithRad) *
                   sin(tSlopeRad) * cos(azimuthRad - tAspectRad))));

    return tHillShade;
}


double Raster::determineContourCurvature(const Eigen::MatrixXd& data,
        int row, int col, double postSpacing, double valueToIgnore)
{

        double mean = 0.0;
        unsigned int nvals = 0;

        double value = static_cast<double>(data(row, col));
        double north = GetNeighbor(data, row, col, NORTH);
        double south = GetNeighbor(data, row, col, SOUTH);
        double east = GetNeighbor(data, row, col, EAST);
        double west = GetNeighbor(data, row, col, WEST);
        double northeast = GetNeighbor(data, row, col, NORTHEAST);
        double northwest = GetNeighbor(data, row, col, NORTHWEST);
        double southeast = GetNeighbor(data, row, col, SOUTHEAST);
        double southwest = GetNeighbor(data, row, col, SOUTHWEST);

        auto accumulate = [&nvals, &mean, valueToIgnore](double val)
        {
            if (val != valueToIgnore)
            {
                mean += val;
                nvals++;
            }
        };

        accumulate(value);
        accumulate(north);
        accumulate(south);
        accumulate(east);
        accumulate(west);
        accumulate(northeast);
        accumulate(northwest);
        accumulate(southeast);
        accumulate(southwest);

        mean /= nvals;

        if (value == valueToIgnore) value = mean;
        if (north == valueToIgnore) north = mean;
        if (south == valueToIgnore) south = mean;
        if (east == valueToIgnore) east = mean;
        if (west == valueToIgnore) west = mean;
        if (northeast == valueToIgnore) northeast = mean;
        if (northwest == valueToIgnore) northwest = mean;
        if (southeast == valueToIgnore) southeast = mean;
        if (southwest == valueToIgnore) southwest = mean;


    double retval = 0.0;

    // Eigen::Matrix3d block = data.block<3,3>(row-1, col-1);
    // int numGood = 0;
    // for (int i = 0; i < 9; ++i)
    // {
    //     if (block(i) != valueToIgnore)
    //         numGood++;
    // }
    //
    // if (numGood > 0)
    // {
    //     Eigen::VectorXd v(numGood);
    //     int goodIdx = 0;
    //     for (int i = 0; i < 9; ++i)
    //     {
    //         if (block(i) != valueToIgnore)
    //             v(goodIdx++) = block(i);
    //     }
    //
    //     for (int i = 0; i < 9; ++i)
    //     {
    //         if (block(i) == valueToIgnore)
    //             block(i) = v.mean();
    //     }
    //
    //     double northwest = block(0);
    //     double west = block(1);
    //     double southwest = block(2);
    //     double north = block(3);
    //     double value = block(4);
    //     double south = block(5);
    //     double northeast = block(6);
    //     double east = block(7);
    //     double southeast = block(8);

        double invSqrSpacing = 1.0 / (postSpacing * postSpacing);
        double invTwiceSpacing = 1.0 / (2 * postSpacing);
        double zXX = (east - 2.0 * value + west) * invSqrSpacing;
        double zYY = (north - 2.0 * value + south) * invSqrSpacing;
        double zXY = ((-1.0 * northwest) + northeast + southwest - southeast) * invSqrSpacing * 0.25;
        double zX = (east - west) * invTwiceSpacing;
        double zY = (north - south) * invTwiceSpacing;
        double p = (zX * zX) + (zY * zY);
        double q = p + 1;

        retval = ((zXX*zX*zX)-(2*zXY*zX*zY)+(zYY*zY*zY))/(p*std::sqrt(q*q*q));
    // }

    return retval;
}


double Raster::determineProfileCurvature(const Eigen::MatrixXd& data, int row,
        int col, double postSpacing,
        double valueToIgnore)
{
    double mean = 0.0;
    unsigned int nvals = 0;

    double value = static_cast<double>(data(row, col));
    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);
    double northeast = GetNeighbor(data, row, col, NORTHEAST);
    double northwest = GetNeighbor(data, row, col, NORTHWEST);
    double southeast = GetNeighbor(data, row, col, SOUTHEAST);
    double southwest = GetNeighbor(data, row, col, SOUTHWEST);

    auto accumulate = [&nvals, &mean, valueToIgnore](double val)
    {
        if (val != valueToIgnore)
        {
            mean += val;
            nvals++;
        }
    };

    accumulate(value);
    accumulate(north);
    accumulate(south);
    accumulate(east);
    accumulate(west);
    accumulate(northeast);
    accumulate(northwest);
    accumulate(southeast);
    accumulate(southwest);

    mean /= nvals;

    if (value == valueToIgnore) value = mean;
    if (north == valueToIgnore) north = mean;
    if (south == valueToIgnore) south = mean;
    if (east == valueToIgnore) east = mean;
    if (west == valueToIgnore) west = mean;
    if (northeast == valueToIgnore) northeast = mean;
    if (northwest == valueToIgnore) northwest = mean;
    if (southeast == valueToIgnore) southeast = mean;
    if (southwest == valueToIgnore) southwest = mean;

    double zXX = (east - 2.0 * value + west) / (postSpacing * postSpacing);
    double zYY = (north - 2.0 * value + south) / (postSpacing * postSpacing);
    double zXY = ((-1.0 * northwest) + northeast + southwest - southeast) / (4.0 * postSpacing * postSpacing);
    double zX = (east - west) / (2 * postSpacing);
    double zY = (north - south) / (2 * postSpacing);
    double p = (zX * zX) + (zY * zY);
    double q = p + 1;

    return static_cast<float>(((zXX*zX*zX)+(2*zXY*zX*zY)+(zYY*zY*zY))/(p*std::sqrt(q*q*q)));
}


double Raster::determineTangentialCurvature(const Eigen::MatrixXd& data,
        int row, int col, double postSpacing, double valueToIgnore)
{
    double mean = 0.0;
    unsigned int nvals = 0;

    double value = static_cast<double>(data(row, col));
    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);
    double northeast = GetNeighbor(data, row, col, NORTHEAST);
    double northwest = GetNeighbor(data, row, col, NORTHWEST);
    double southeast = GetNeighbor(data, row, col, SOUTHEAST);
    double southwest = GetNeighbor(data, row, col, SOUTHWEST);

    auto accumulate = [&nvals, &mean, valueToIgnore](double val)
    {
        if (val != valueToIgnore)
        {
            mean += val;
            nvals++;
        }
    };

    accumulate(value);
    accumulate(north);
    accumulate(south);
    accumulate(east);
    accumulate(west);
    accumulate(northeast);
    accumulate(northwest);
    accumulate(southeast);
    accumulate(southwest);

    mean /= nvals;

    if (value == valueToIgnore) value = mean;
    if (north == valueToIgnore) north = mean;
    if (south == valueToIgnore) south = mean;
    if (east == valueToIgnore) east = mean;
    if (west == valueToIgnore) west = mean;
    if (northeast == valueToIgnore) northeast = mean;
    if (northwest == valueToIgnore) northwest = mean;
    if (southeast == valueToIgnore) southeast = mean;
    if (southwest == valueToIgnore) southwest = mean;

    double zXX = (east - 2.0 * value + west) / (postSpacing * postSpacing);
    double zYY = (north - 2.0 * value + south) / (postSpacing * postSpacing);
    double zXY = ((-1.0 * northwest) + northeast + southwest - southeast) / (4.0 * postSpacing * postSpacing);
    double zX = (east - west) / (2 * postSpacing);
    double zY = (north - south) / (2 * postSpacing);
    double p = (zX * zX) + (zY * zY);
    double q = p + 1;

    return static_cast<float>(((zXX*zY*zY)-(2*zXY*zX*zY)+(zYY*zX*zX))/(p*std::sqrt(q)));
}


double Raster::determineTotalCurvature(const Eigen::MatrixXd& data, int row,
                                       int col, double postSpacing,
                                       double valueToIgnore)
{
    double mean = 0.0;
    unsigned int nvals = 0;

    double value = static_cast<double>(data(row, col));
    double north = GetNeighbor(data, row, col, NORTH);
    double south = GetNeighbor(data, row, col, SOUTH);
    double east = GetNeighbor(data, row, col, EAST);
    double west = GetNeighbor(data, row, col, WEST);
    double northeast = GetNeighbor(data, row, col, NORTHEAST);
    double northwest = GetNeighbor(data, row, col, NORTHWEST);
    double southeast = GetNeighbor(data, row, col, SOUTHEAST);
    double southwest = GetNeighbor(data, row, col, SOUTHWEST);

    auto accumulate = [&nvals, &mean, valueToIgnore](double val)
    {
        if (val != valueToIgnore)
        {
            mean += val;
            nvals++;
        }
    };

    accumulate(value);
    accumulate(north);
    accumulate(south);
    accumulate(east);
    accumulate(west);
    accumulate(northeast);
    accumulate(northwest);
    accumulate(southeast);
    accumulate(southwest);

    mean /= nvals;

    if (value == valueToIgnore) value = mean;
    if (north == valueToIgnore) north = mean;
    if (south == valueToIgnore) south = mean;
    if (east == valueToIgnore) east = mean;
    if (west == valueToIgnore) west = mean;
    if (northeast == valueToIgnore) northeast = mean;
    if (northwest == valueToIgnore) northwest = mean;
    if (southeast == valueToIgnore) southeast = mean;
    if (southwest == valueToIgnore) southwest = mean;

    double zXX = (east - 2.0 * value + west) / (postSpacing * postSpacing);
    double zYY = (north - 2.0 * value + south) / (postSpacing * postSpacing);
    double zXY = ((-1.0 * northwest) + northeast + southwest - southeast) / (4.0 * postSpacing * postSpacing);

    return static_cast<float>((zXX * zXX) + (2.0 * zXY * zXY) + (zYY * zYY));
}


GDALDataset* Raster::createFloat32GTIFF(std::string const filename, int rows,
                                        int cols)
{
    char **papszMetadata;

    // parse the format driver, hardcoded for the time being
    std::string tFormat("GTIFF");
    const char *pszFormat = tFormat.c_str();
    GDALDriver* tpDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);

    // try to create a file of the requested format
    if (tpDriver != NULL)
    {
        papszMetadata = tpDriver->GetMetadata();
        if (CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE))
        {
            char **papszOptions = NULL;

            boost::filesystem::path p(filename);
            p.replace_extension(".tif");
            GDALDataset *dataset;
            dataset = tpDriver->Create(p.string().c_str(), cols, rows, 1,
                                       GDT_Float32, papszOptions);

            BOX2D& extent = getBounds();

            // set the geo transformation
            double adfGeoTransform[6];
            adfGeoTransform[0] = extent.minx; // - 0.5*m_spacing_x;
            adfGeoTransform[1] = m_spacing_x;
            adfGeoTransform[2] = 0.0;
            adfGeoTransform[3] = extent.maxy; // + 0.5*m_spacing_y;
            adfGeoTransform[4] = 0.0;
            adfGeoTransform[5] = -1 * m_spacing_y;
            dataset->SetGeoTransform(adfGeoTransform);

            // set the projection
            m_log->get(LogLevel::Debug5) << m_inSRS.getWKT() << std::endl;
            dataset->SetProjection(m_inSRS.getWKT().c_str());

            if (dataset)
                return dataset;
        }
    }
    //ABELL
    return NULL;
}

void Raster::finalizeFloat32GTIFF(GDALDataset* dataset, float* raster, int cols, int rows, double background)
{
  GDALRasterBand *tBand = dataset->GetRasterBand(1);

  tBand->SetNoDataValue((double)background);

  if (rows > 0 && cols > 0)
      tBand->RasterIO(GF_Write, 0, 0, cols, rows,
                      raster, cols, rows,
                      GDT_Float32, 0, 0);
}


void Raster::writeSlope(std::string const filename, SlopeMethod method)
{
    // use the max grid size as the post spacing
    double tPostSpacing = std::max(m_spacing_x, m_spacing_y);

    m_log->get(LogLevel::Debug5) << tPostSpacing << std::endl;

    GDALDataset *mpDstDS;
    mpDstDS = createFloat32GTIFF(filename, m_rows, m_cols);

    if (mpDstDS)
    {
        // loop over the raster and determine max slope at each location
        int tXStart = 1, tXEnd = m_cols - 1;
        int tYStart = 1, tYEnd = m_rows - 1;
        float *poRasterData = new float[m_rows*m_cols];
        for (uint32_t i=0; i<m_rows*m_cols; i++)
            poRasterData[i] = m_background;

        m_log->get(LogLevel::Debug) << "Calculating slope... ";
        auto start = std::chrono::high_resolution_clock::now();
        for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        {
            int tXIn = tXOut;
            for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
            {
                int tYIn = tYOut;

                float tSlopeValDegree(0);

                switch (method)
                {
                    case SD8:
                        tSlopeValDegree = (float)determineSlopeD8(m_dem,
                                          tYOut, tXOut, tPostSpacing,
                                          m_background);
                        break;

                    case SFD:
                        tSlopeValDegree = (double)determineSlopeFD(m_dem,
                                          tYOut, tXOut, tPostSpacing,
                                          m_background);
                        break;
                }

                poRasterData[(tYIn * m_cols) + tXIn] =
                    std::tan(tSlopeValDegree*c_pi/180.0)*100.0;
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        m_log->get(LogLevel::Debug) << "done in " << GetTime(start, stop) << " ms\n";

        if (poRasterData)
            finalizeFloat32GTIFF(mpDstDS, poRasterData, m_cols, m_rows, m_background);

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}


void Raster::writeAspect(std::string const filename, AspectMethod method)
{
    // use the max grid size as the post spacing
    double tPostSpacing = std::max(m_spacing_x, m_spacing_y);

    GDALDataset *mpDstDS;
    mpDstDS = createFloat32GTIFF(filename, m_rows, m_cols);

    if (mpDstDS)
    {
        // loop over the raster and determine max slope at each location
        int tXStart = 1, tXEnd = m_cols - 1;
        int tYStart = 1, tYEnd = m_rows - 1;
        float *poRasterData = new float[m_rows*m_cols];
        for (uint32_t i=0; i<m_rows*m_cols; i++)
            poRasterData[i] = 0;    // Initialize all elements to zero.

        for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        {
            int tXIn = tXOut;
            for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
            {
                int tYIn = tYOut;

                float tSlopeValDegree(0);

                //Compute Aspect Value
                switch (method)
                {
                    case AD8:
                        tSlopeValDegree = (float)determineAspectD8(m_dem,
                                          tYOut, tXOut, tPostSpacing);
                        break;

                    case SFD:
                        tSlopeValDegree = (float)determineAspectFD(m_dem,
                                          tYOut, tXOut, tPostSpacing, m_background);
                        break;
                }

                if (tSlopeValDegree == std::numeric_limits<double>::max())
                    poRasterData[(tYIn * m_cols) + tXIn] = m_background;
                else
                    poRasterData[(tYIn * m_cols) + tXIn] = tSlopeValDegree;
            }
        }

        if (poRasterData)
            finalizeFloat32GTIFF(mpDstDS, poRasterData, m_cols, m_rows, m_background);

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}


void Raster::writeCatchmentArea(std::string const filename)
{
    Eigen::MatrixXd area(m_cols, m_rows);
    area.setZero();

    // use the max grid size as the post spacing
    double tPostSpacing = std::max(m_spacing_x, m_spacing_y);

    GDALDataset *mpDstDS;
    mpDstDS = createFloat32GTIFF(filename, m_rows, m_cols);

    if (mpDstDS)
    {
        // loop over the raster and determine max slope at each location
        int tXStart = 1, tXEnd = m_cols - 1;
        int tYStart = 1, tYEnd = m_rows - 1;
        float *poRasterData = new float[m_rows*m_cols];
        for (uint32_t i=0; i<m_rows*m_cols; i++)
            poRasterData[i] = m_background;    // Initialize all elements to zero.


        int tXOut = tXStart;
        int tYOut = tYStart;
        //for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        //{
        //    for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
        //    {
        //Compute Aspect Value
        //switch (method)
        //{
        //case AD8:
        //tSlopeValDegree = (float)determineAspectD8(m_dem, tYOut, tXOut, tPostSpacing);
        //break;
        //
        //case SFD:
        //  tSlopeValDegree = (float)determineAspectFD(m_dem, tYOut, tXOut, tPostSpacing, m_background);
        //break;
        //}
        area(tYOut, tXOut) =
            determineCatchmentAreaD8(m_dem, area, tYOut, tXOut, tPostSpacing);
        //    }
        // }

        #pragma omp parallel for
        for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        {
            for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
            {
                poRasterData[(tYOut * m_cols) + tXOut] = area(tYOut, tXOut);
            }
        }

        if (poRasterData)
            finalizeFloat32GTIFF(mpDstDS, poRasterData, m_cols, m_rows, m_background);

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}

void Raster::writeHillshade(std::string const filename)
{
    // use the max grid size as the post spacing
    double tPostSpacing = std::max(m_spacing_x, m_spacing_y);

    GDALDataset *mpDstDS;
    mpDstDS = createFloat32GTIFF(filename, m_rows, m_cols);

    if (mpDstDS)
    {
        int tXStart = 1, tXEnd = m_cols - 1;
        int tYStart = 1, tYEnd = m_rows - 1;
        float *poRasterData = new float[m_rows*m_cols];
        for (uint32_t i=0; i<m_rows*m_cols; i++)
            poRasterData[i] = 0;    // Initialize all elements to zero.

        // Parameters for hill shade
        double illumAltitudeDegree = 45.0;
        double illumAzimuthDegree = 315.0;
        double tZenithRad = (90 - illumAltitudeDegree) * (c_pi / 180.0);
        double tAzimuthMath = 360.0 - illumAzimuthDegree + 90;

        if (tAzimuthMath >= 360.0)
            tAzimuthMath = tAzimuthMath - 360.0;

        double tAzimuthRad = tAzimuthMath * (c_pi / 180.0);

        double min_val = std::numeric_limits<double>::max();
        double max_val = -std::numeric_limits<double>::max();

        for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        {
            for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
            {
                float tSlopeValDegree = (float)determineHillshade(m_dem,
                                        tYOut, tXOut, tZenithRad, tAzimuthRad,
                                        tPostSpacing);

                if (tSlopeValDegree == std::numeric_limits<double>::max())
                    poRasterData[(tYOut * m_cols) + tXOut] = m_background;
                else
                    poRasterData[(tYOut * m_cols) + tXOut] = tSlopeValDegree;

                if (tSlopeValDegree < min_val) min_val = tSlopeValDegree;
                if (tSlopeValDegree > max_val) max_val = tSlopeValDegree;
            }
        }

        if (poRasterData)
            finalizeFloat32GTIFF(mpDstDS, poRasterData, m_cols, m_rows,
                                 m_background);

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}


void Raster::writeCurvature(std::string const filename, CurvatureType curveType,
                            double valueToIgnore)
{
    // use the max grid size as the post spacing
    double tPostSpacing = std::max(m_spacing_x, m_spacing_y);

    GDALDataset *mpDstDS;
    mpDstDS = createFloat32GTIFF(filename, m_rows, m_cols);

    if (mpDstDS)
    {
        int tXStart = 1, tXEnd = m_cols - 1;
        int tYStart = 1, tYEnd = m_rows - 1;
        float *poRasterData = new float[m_rows*m_cols];
        for (uint32_t i=0; i<m_rows*m_cols; i++)
            poRasterData[i] = m_background;

        m_log->get(LogLevel::Debug) << "Calculating curvature... ";
        auto start = std::chrono::high_resolution_clock::now();
        for (int tXOut = tXStart; tXOut < tXEnd; tXOut++)
        {
            int tXIn = tXOut;
            for (int tYOut = tYStart; tYOut < tYEnd; tYOut++)
            {
                int tYIn = tYOut;

                double curve(0);

                switch (curveType)
                {
                    case CONTOUR:
                        curve =
                            determineContourCurvature(m_dem, tYOut, tXOut,
                                                      tPostSpacing,
                                                      m_background);
                        break;

                    case PROFILE:
                        curve = determineProfileCurvature(m_dem,
                                                          tYOut, tXOut,
                                                          tPostSpacing,
                                                          m_background);
                        break;

                    case TANGENTIAL:
                        curve = determineTangentialCurvature(m_dem,
                                                             tYOut, tXOut,
                                                             tPostSpacing,
                                                             m_background);
                        break;

                    case TOTAL:
                        curve = determineTotalCurvature(m_dem,
                                                        tYOut, tXOut,
                                                        tPostSpacing,
                                                        m_background);
                        break;
                }

                poRasterData[(tYOut * m_cols) + tXOut] = static_cast<float>(curve);
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        m_log->get(LogLevel::Debug) << "done in " << GetTime(start, stop) << " ms\n";

        if (poRasterData)
            finalizeFloat32GTIFF(mpDstDS, poRasterData, m_cols, m_rows,
                                 m_background);

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}


} // namespace Utils
} // namespace pdal
