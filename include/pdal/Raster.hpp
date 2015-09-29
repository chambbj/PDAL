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

#include <pdal/pdal_export.hpp>

#include <cstdint>

#include <Eigen/Core>

#include "gdal_priv.h" // For File I/O

#include <pdal/Log.hpp>
#include <pdal/PointView.hpp>

namespace pdal
{

class BOX2D;
class SpatialReference;

namespace Utils
{

enum SlopeMethod
{
    SD8,
    SFD
};

enum AspectMethod
{
    AD8,
    AFD
};

enum PrimitiveType
{
    SLOPE_D8,
    SLOPE_FD,
    ASPECT_D8,
    ASPECT_FD,
    HILLSHADE,
    CONTOUR_CURVATURE,
    PROFILE_CURVATURE,
    TANGENTIAL_CURVATURE,
    TOTAL_CURVATURE,
    CATCHMENT_AREA
};

enum CurvatureType
{
    CONTOUR,
    PROFILE,
    TANGENTIAL,
    TOTAL
};

enum Direction
{
    NORTH,
    SOUTH,
    EAST,
    WEST,
    NORTHEAST,
    NORTHWEST,
    SOUTHEAST,
    SOUTHWEST
};

/** \brief Create surface model (max Z) for the input point cloud.
  * \param tDemData The output raster. Caller must initialize this.
  * \param tDensityData The output raster. Caller must initialize this.
  * \param tMeanData The output raster. Caller must initialize this.
  * \param tStddevData The output raster. Caller must initialize this.
  * \param data The input point cloud.
  * \param spacing_x The post spacing in X.
  * \param spacing_y The post spacing in Y.
  * \param rows The number of rows in the output raster.
  * \param cols The number of columns in the output raster.
  * \param background The initial value in the output raster.
  * \param extent The 2D extents of the input point cloud.
  */
PDAL_DLL void CreateSurface(Eigen::MatrixXd& tDemData,
                            Eigen::MatrixXd& tDensityData,
                            Eigen::MatrixXd& tMeanData,
                            Eigen::MatrixXd& tStddevData,
                            const PointViewPtr data, double spacing_x,
                            double spacing_y, uint32_t rows, uint32_t cols,
                            float background, BOX2D extent);

class PDAL_DLL Raster
{
public:
    Raster::Raster(const PointViewPtr data, double m_spacing_x,
                   double m_spacing_y, SpatialReference inSRS,
                   float c_background, LogPtr log);

    void setBounds(const BOX2D& v)
    {
        m_bounds = v;
    }

    BOX2D& getBounds()
    {
        return m_bounds;
    }

    /** \brief Determine maximum slope at the given row/column using the finite
      * distance method.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineSlopeFD(const Eigen::MatrixXd& data, int row, int col,
                            double postSpacing, double valueToIgnore);

    /** \brief Determine maximum slope at the given row/column using the D8
      * method.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineSlopeD8(const Eigen::MatrixXd& data, int row, int col,
                            double postSpacing, double valueToIgnore);

    /** \brief Determine aspect at the given row/column using the finite
      * distance method.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineAspectFD(const Eigen::MatrixXd& data, int row, int col,
                             double postSpacing, double valueToIgnore);

    /** \brief Determine aspect at the given row/column using the D8 method.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineAspectD8(const Eigen::MatrixXd& data, int row, int col,
                             double postSpacing);

    /** \brief Determine catchment area at the given row/column using the D8
      * method.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    int determineCatchmentAreaD8(const Eigen::MatrixXd& data,
                                 Eigen::MatrixXd& area, int row, int col,
                                 double postSpacing);

    /** \brief Determine contour curvature at the given row/column.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineContourCurvature(const Eigen::MatrixXd& data, int row,
                                     int col, double postSpacing,
                                     double valueToIgnore);

    /** \brief Determine profile curvature at the given row/column.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineProfileCurvature(const Eigen::MatrixXd& data, int row,
                                     int col, double postSpacing,
                                     double valueToIgnore);

    /** \brief Determine tangential curvature at the given row/column.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineTangentialCurvature(const Eigen::MatrixXd& data, int row,
                                        int col, double postSpacing,
                                        double valueToIgnore);

    /** \brief Determine total curvature at the given row/column.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param postSpacing The post spacing used to compute derivatives.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    double determineTotalCurvature(const Eigen::MatrixXd& data, int row,
                                   int col, double postSpacing,
                                   double valueToIgnore);

    /** \brief Determine hillshade at the given row/column.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param zenithRad The zenith in radians.
      * \param azimuthRad The azimuth in radians.
      * \param postSpacing The post spacing used to compute derivatives.
      */
    double determineHillshade(const Eigen::MatrixXd& data, int row, int col,
                              double zenithRad, double azimuthRad,
                              double postSpacing);

    /** \brief Get neighbor's value at the given row/column, in the specified
      * direction.
      * \param data The input raster.
      * \param row The row of the current point.
      * \param col The column of the current point.
      * \param d Direction specifying one of the eight neighors.
      */
    double GetNeighbor(const Eigen::MatrixXd& data, int row, int col,
                       Direction d);

    /** \brief Write slope raster to the given filename, using the specified
      * method.
      * \param filename The slope raster filename to write.
      * \param method The slope method (default: D8).
      */
    void writeSlope(std::string const filename, SlopeMethod method=SD8);
    void writeDensity(std::string const filename);
    void writeMean(std::string const filename);
    void writeStddev(std::string const filename);
    void writeDEMCutoff(std::string const filename);

    /** \brief Write aspect raster to the given filename, using the specified
      * method.
      * \param filename The aspect raster filename to write.
      * \param method The aspect method (default: D8).
      */
    void writeAspect(std::string const filename, AspectMethod method=AD8);

    /** \brief Write catchment area raster to the given filename.
      * \param filename The catchment area raster filename to write.
      */
    void writeCatchmentArea(std::string const filename);

    /** \brief Write hillshade raster to the given filename.
      * \param filename The hillshade raster filename to write.
      */
    void writeHillshade(std::string const filename);

    /** \brief Write curvature raster to the given filename, using the
      * specified method.
      * \param filename The curvature raster filename to write.
      * \param method The curvature method.
      * \param valueToIgnore The value representing an empty cell - to be ignored.
      */
    void writeCurvature(std::string const filename, CurvatureType curveType,
                        double valueToIgnore);

    /** \brief Create a 32-bit floating point GeoTIFF.
      * \param filename The raster filename to write.
      * \param cols The number of columns in the new GeoTIFF.
      * \param rows The number of rows in the new GeoTIFF.
      */
    GDALDataset* createFloat32GTIFF(std::string const filename, int cols,
                                    int rows);

    /** \brief Calculate number of rows/columns required for the output raster
      * at the given X, Y post spacing.
      */
    void calculateGridSizes();

    void finalizeFloat32GTIFF(GDALDataset* dataset, float* raster, int cols, int rows, double background);

private:
    BOX2D m_bounds;
    Eigen::MatrixXd m_dem;
    Eigen::MatrixXd m_density;
    Eigen::MatrixXd m_mean;
    Eigen::MatrixXd m_stddev;
    double m_spacing_x;
    double m_spacing_y;
    uint32_t m_rows;
    uint32_t m_cols;
    double m_background;
    SpatialReference m_inSRS;
    LogPtr m_log;

    Raster& operator=(const Raster&); // not implemented
    Raster(const Raster&); // not implemented
};

} // namespace Utils
} // namespace pdal
