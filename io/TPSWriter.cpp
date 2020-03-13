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

#include <pdal/EigenUtils.hpp>
#include <pdal/GDALUtils.hpp>
#include <pdal/KDIndex.hpp>
#include <pdal/PointView.hpp>

#include "../filters/private/delaunator.hpp"

#include <Eigen/Dense>

namespace pdal
{

namespace
{
// Copy/paste from HagDelaunay.cpp

// https://en.wikipedia.org/wiki/Barycentric_coordinate_system
// http://blackpawn.com/texts/pointinpoly/default.html
//
// If x/y is in the triangle, we'll return a valid distance.
// If not, we return infinity.  If the determinant is 0, the input points
// aren't a triangle (they're collinear).
double distance_along_z(double x1, double y1, double z1, double x2, double y2,
                        double z2, double x3, double y3, double z3, double x,
                        double y)
{
    double z = std::numeric_limits<double>::infinity();

    double detT = ((y2 - y3) * (x1 - x3)) + ((x3 - x2) * (y1 - y3));

    // ABELL - should probably check something close to 0, rather than
    // exactly 0.
    if (detT != 0.0)
    {
        // Compute the barycentric coordinates of x,y (relative to
        // x1/y1, x2/y2, x3/y3).  Essentially the weight that each
        // corner of the triangle contributes to the point in question.

        // Another way to think about this is that we're making a basis
        // for the system with the basis vectors being two sides of
        // the triangle.  You can rearrange the z calculation below in
        // terms of lambda1 and lambda2 to see this.  Also note that
        // since lambda1 and lambda2 are coefficients of the basis vectors,
        // any values outside of the range [0,1] are necessarily out of the
        // triangle.
        double lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT;
        double lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT;
        if (lambda1 >= 0 && lambda1 <= 1 && lambda2 >= 0 && lambda2 <= 1)
        {
            double sum = lambda1 + lambda2;
            if (sum <= 1)
            {
                double lambda3 = 1 - sum;
                z = (lambda1 * z1) + (lambda2 * z2) + (lambda3 * z3);
            }
        }
    }
    return z;
}

// The non-ground point (x0, y0) is in exactly 0 or 1 of the triangles of
// the ground triangulation, so when we find a triangle containing the point,
// return the interpolated z.
// (I suppose the point could be on a edge of two triangles, but the
//  result is the same, so this is still good.)
double delaunay_interp_ground(double x0, double y0, PointViewPtr gView,
                              const PointIdList& ids)
{
    using namespace pdal::Dimension;

    // Delaunay-based interpolation
    std::vector<double> neighbors;

    for (size_t j = 0; j < ids.size(); ++j)
    {
        neighbors.push_back(gView->getFieldAs<double>(Id::X, ids[j]));
        neighbors.push_back(gView->getFieldAs<double>(Id::Y, ids[j]));
    }

    delaunator::Delaunator triangulation(neighbors);
    const std::vector<size_t>& triangles(triangulation.triangles);

    for (size_t j = 0; j < triangles.size(); j += 3)
    {
        auto ai = triangles[j + 0];
        auto bi = triangles[j + 1];
        auto ci = triangles[j + 2];
        double ax = gView->getFieldAs<double>(Id::X, ids[ai]);
        double ay = gView->getFieldAs<double>(Id::Y, ids[ai]);
        double az = gView->getFieldAs<double>(Id::Z, ids[ai]);

        double bx = gView->getFieldAs<double>(Id::X, ids[bi]);
        double by = gView->getFieldAs<double>(Id::Y, ids[bi]);
        double bz = gView->getFieldAs<double>(Id::Z, ids[bi]);

        double cx = gView->getFieldAs<double>(Id::X, ids[ci]);
        double cy = gView->getFieldAs<double>(Id::Y, ids[ci]);
        double cz = gView->getFieldAs<double>(Id::Z, ids[ci]);

        // Returns infinity unless the point x0/y0 is in the triangle.
        double z1 =
            distance_along_z(ax, ay, az, bx, by, bz, cx, cy, cz, x0, y0);
        if (z1 != std::numeric_limits<double>::infinity())
            return z1;
    }
    // If the non ground point was outside the triangulation of ground
    // points, just use the Z coordinate of the closest
    // ground point.
    return gView->getFieldAs<double>(Id::Z, ids[0]);
}

} // unnamed namespace

static StaticPluginInfo const s_info{
    "writers.tin",
    "Write a GDAL raster interpolated from a TPS.",
    "http://pdal.io/stages/writers.tin.html",
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
    using namespace pdal::Dimension;

    PointViewPtr gView = view->makeNew();

    // Separate into ground and non-ground views.
    for (PointRef const& point : *view)
    {
        if (point.getFieldAs<uint8_t>(Id::Classification) == ClassLabel::Ground)
            gView->appendPoint(*view, point.pointId());
    }
    BOX2D gBounds;
    gView->calculateBounds(gBounds);

    // Bail if there weren't any points classified as ground.
    if (gView->size() == 0)
        throwError("Input PointView does not have any points classified "
                   "as ground");

    std::vector<long> cols(((gBounds.maxx - gBounds.minx) / m_edgeLength) + 1);
    std::iota(cols.begin(), cols.end(), 0);
    std::vector<long> rows(((gBounds.maxy - gBounds.miny) / m_edgeLength) + 1);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> z1(rows.size() * cols.size(),
                           std::numeric_limits<double>::quiet_NaN());

    // Build the 2D KD-tree.
    const KD2Index& kdi = gView->build2dIndex();

    // determine number of rows/cols, create the output matrix, iterate
    for (long const& row : rows)
    {
        for (long const& col : cols)
        {
            double x0 = gBounds.minx + (col + 0.5) * m_edgeLength;
            double y0 = gBounds.miny + (row + 0.5) * m_edgeLength;
            PointIdList ids(m_count);
            std::vector<double> sqr_dists(m_count);
            kdi.knnSearch(x0, y0, m_count, &ids, &sqr_dists);

            // Closest ground point.
            PointRef gNearest = gView->point(ids[0]);
            double x = gNearest.getFieldAs<double>(Id::X);
            double y = gNearest.getFieldAs<double>(Id::Y);
            double z = gNearest.getFieldAs<double>(Id::Z);

            // If the close ground point is at the same X/Y as the non-ground
            // point, we're done.  Also, if there's only one ground point, we
            // just use that.
            if ((x0 == x && y0 == y) || ids.size() == 1)
            {
                z1[col * rows.size() + row] = z;
            }
            // If the non-ground point is outside the bounds of all the
            // ground points and we're not doing extrapolation, just return
            // its current Z, which will give a HAG of 0.
            else if (
                !gBounds.contains(
                    x0, y0) /* && !m_allowExtrapolation*/) // we currently have
                                                           // no
                                                           // allowExtrapolation
                                                           // could bring over
                                                           // from HagDelaunay
            {
                // z1[col*rows.size()+row] = z0;
                // but the raster has no z0, maybe z again? or leave as NaN?
                z1[col * rows.size() + row] = z;
            }
            else
            {
                z1[col * rows.size() + row] =
                    delaunay_interp_ground(x0, y0, gView, ids);
            }
        }
    }
    Eigen::MatrixXd zInterp =
        Eigen::Map<Eigen::MatrixXd>(z1.data(), rows.size(), cols.size());
    writeMatrix(zInterp, m_filename, "GTiff", m_edgeLength, gBounds,
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
