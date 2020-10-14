/******************************************************************************
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
 *     * Neither the name of Hobu, Inc. nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
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

#include "TPSWriter.hpp"

#include <numeric>

#include <pdal/GDALUtils.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/PointView.hpp>
#include <pdal/private/MathUtils.hpp>

#include <Eigen/Dense>

namespace pdal
{

using namespace Dimension;
using namespace Eigen;

namespace
{

// The non-ground point (x0, y0) is in exactly 0 or 1 of the triangles of
// the ground triangulation, so when we find a triangle containing the point,
// return the interpolated z.
// (I suppose the point could be on a edge of two triangles, but the
//  result is the same, so this is still good.)
double delaunay_interp_ground(double x0, double y0, PointViewPtr gView,
                              point_count_t count)
{
    // Build the 2D KD-tree.
    const KD2Index& gIndex = gView->build2dIndex();

    auto CalcRbfValue = [](const VectorXd& xi, const VectorXd& xj) {
        double r = (xj - xi).norm();
        double value = r * r * std::log(r);
        // std::cerr << "r: " << r << ", rbf: " << value << std::endl;
        return std::isnan(value) ? 0.0 : value;
    };
    PointIdList gIds = gIndex.neighbors(x0, y0, count);
    MatrixXd X = MatrixXd::Zero(2, count);
    VectorXd y = VectorXd::Zero(count);
    VectorXd x = VectorXd::Zero(2);
    for (point_count_t i = 0; i < count; ++i)
    {
        X(0, i) = gView->getFieldAs<double>(Id::X, gIds[i]);
        X(1, i) = gView->getFieldAs<double>(Id::Y, gIds[i]);
        y(i) = gView->getFieldAs<double>(Id::Z, gIds[i]);
    }
    x(0) = x0;
    x(1) = y0;
    MatrixXd Phi = MatrixXd::Zero(count, count);
    for (point_count_t i = 0; i < count; ++i)
    {
        for (point_count_t j = 0; j < count; ++j)
        {
            double value = CalcRbfValue(X.col(i), X.col(j));
            Phi(i, j) = Phi(j, i) = value;
        }
    }
    /*
    MatrixXd A = m_lambdaArg->set()
                     ? Phi.transpose() * Phi +
                           m_lambda * MatrixXd::Identity(count, count)
                     : Phi;
    VectorXd b = m_lambdaArg->set() ? Phi.transpose() * y : y;
    VectorXd w = Eigen::PartialPivLU<MatrixXd>(A).solve(b);
    */
    VectorXd w = Eigen::PartialPivLU<MatrixXd>(Phi).solve(y);

    double val(0.0);
    for (point_count_t i = 0; i < count; ++i)
    {
        val += w(i) * CalcRbfValue(x, X.col(i));
    }
    return val;
}

} // unnamed namespace

static StaticPluginInfo const s_info{
    "writers.tps",
    "Write a GDAL raster interpolated from a TPS.",
    "http://pdal.io/stages/writers.tps.html",
    {"tif", "tiff"}};

CREATE_STATIC_STAGE(TPSWriter, s_info)

std::string TPSWriter::getName() const
{
    return s_info.name;
}

void TPSWriter::addArgs(ProgramArgs& args)
{
    args.add("filename", "Output filename", m_filename).setPositional();
    args.add("resolution", "Cell edge size, in units of X/Y", m_edgeLength)
        .setPositional();
    args.add("gdaldriver", "TPS writer driver name", m_drivername, "GTiff");
    args.add("gdalopts", "TPS driver options (name=value,name=value...)",
             m_options);
    args.add("data_type",
             "Data type for output grid (\"int8\", \"uint64\", "
             "\"float\", etc.)",
             m_dataType, Dimension::Type::Double);
    // Nan is a sentinal value to say that no value was set for nodata.
    args.add("nodata", "No data value", m_noData,
             std::numeric_limits<double>::quiet_NaN());
    // m_xOriginArg = &args.add("origin_x", "X origin for grid.", m_xOrigin);
    // m_yOriginArg = &args.add("origin_y", "Y origin for grid.", m_yOrigin);
    // m_widthArg = &args.add("width", "Number of cells in the X direction.",
    //    m_width);
    // m_heightArg = &args.add("height", "Number of cells in the Y direction.",
    //    m_height);
    args.add("count",
             "The number of points to fetch to determine the "
             "ground point [Default: 10].",
             m_count, point_count_t(10));
}

void TPSWriter::initialize()
{
    /*
        int args = 0;
        if (m_xOriginArg->set())
            args |= 1;
        if (m_yOriginArg->set())
            args |= 2;
        if (m_heightArg->set())
            args |= 4;
        if (m_widthArg->set())
            args |= 8;
        if (args != 0 && args != 15)
            throwError("Must specify all or none of 'origin_x', 'origin_y', "
                "'width' and 'height'.");
        if (args == 15)
        {
            if (m_bounds.to2d().valid())
                throwError("Specify either 'bounds' or 'origin_x'/'origin_y'/"
                    "'width'/'height' options -- not both");

            // Subtracting .5 gets to the middle of the last cell.  This
            // should get us back to the same place when figuring the
            // cell count.
            m_bounds = Bounds({m_xOrigin, m_yOrigin,
                m_xOrigin + (m_edgeLength * (m_width - .5)),
                m_yOrigin + (m_edgeLength * (m_height - .5))});
        }

        m_fixedGrid = m_bounds.to2d().valid();
        // If we've specified a grid, we don't expand by point.  We also
        // don't expand by point if we're running in standard mode.  That's
        // set later in writeView.
        m_expandByPoint = !m_fixedGrid;
    */
    gdal::registerDrivers();
}

void TPSWriter::write(const PointViewPtr view)
{
    // Create a fixed grid, based off bounds, origin, height/width, and cell
    // size considerations. Scan all locations in the fixed grid, obtaining XY.
    // Interpolate Z based off point cloud (all points/ground only).

    PointViewPtr gView = view->makeNew();

    // Separate into ground and non-ground views.
    for (PointRef const& point : *view)
    {
        if (point.getFieldAs<uint8_t>(Id::Classification) == ClassLabel::Ground)
            gView->appendPoint(*view, point.pointId());
    }

    // Bail if there weren't any points classified as ground.
    if (gView->size() == 0)
        throwError("Input PointView does not have any points classified "
                   "as ground");

    BOX2D bounds;
    view->calculateBounds(bounds);

    std::vector<long> cols(((bounds.maxx - bounds.minx) / m_edgeLength) + 1);
    std::iota(cols.begin(), cols.end(), 0);
    std::vector<long> rows(((bounds.maxy - bounds.miny) / m_edgeLength) + 1);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> z1(rows.size() * cols.size(),
                           std::numeric_limits<double>::quiet_NaN());

    // determine number of rows/cols, create the output matrix, iterate
    for (long const& row : rows)
    {
        for (long const& col : cols)
        {
            double x0 = bounds.minx + (col + 0.5) * m_edgeLength;
            double y0 = bounds.miny + (row + 0.5) * m_edgeLength;

            z1[col * rows.size() + row] =
                delaunay_interp_ground(x0, y0, gView, m_count);
        }
    }
    MatrixXd zInterp =
        Eigen::Map<MatrixXd>(z1.data(), rows.size(), cols.size());
    math::writeMatrix(zInterp, m_filename, "GTiff", m_edgeLength, bounds,
                      gView->spatialReference());
}

void TPSWriter::done(PointTableRef table)
{
    /*
        std::array<double, 6> pixelToPos;

        pixelToPos[0] = m_origin.x;
        pixelToPos[1] = m_edgeLength;
        pixelToPos[2] = 0;
        pixelToPos[3] = m_origin.y + (m_edgeLength * m_grid->height());
        pixelToPos[4] = 0;
        pixelToPos[5] = -m_edgeLength;
        gdal::Raster raster(m_outputFilename, m_drivername, m_srs, pixelToPos);

        gdal::GDALError err = raster.open(m_grid->width(), m_grid->height(),
            m_grid->numBands(), m_dataType, m_noData, m_options);

        if (err != gdal::GDALError::None)
            throwError(raster.errorMsg());
        int bandNum = 1;

        double *src;
        src = m_grid->data("min");
        double srcNoData = std::numeric_limits<double>::quiet_NaN();
        if (src && err == gdal::GDALError::None)
            err = raster.writeBand(src, srcNoData, bandNum++, "min");
        if (err != gdal::GDALError::None)
            throwError(raster.errorMsg());
    */

    getMetadata().addList("filename", m_filename);
}

} // namespace pdal
