/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
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

/*
 * 1) Initialization
 *
 *    Open at window size of 11
 *    Close at window size of 9
 *    Replace points with z'-z >= 1m
 *
 * 2) Control point selection
 *
 *    Form n levels of hierarchy, with lowest elevation points from the
 *    initialization step seeding the grids.
 *
 * 3) Thin plate spline surface interpolation
 * 4) Filtering
 */

#include "MongusFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Dense>

#include <unordered_map>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.mongus", "Progressive morphological filter",
               "http://pdal.io/stages/filters.mongus.html");

CREATE_STATIC_PLUGIN(1, 0, MongusFilter, Filter, s_info)

std::string MongusFilter::getName() const
{
    return s_info.name;
}


void MongusFilter::addArgs(ProgramArgs& args)
{
    args.add("classify", "Apply classification labels?", m_classify, true);
    args.add("extract", "Extract ground returns?", m_extract);
}


void MongusFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
}

std::vector<double> MongusFilter::morphOpen(PointViewPtr view, int radius)
{
    point_count_t np(view->size());

    auto hash = calculateHash(view);
    log()->get(LogLevel::Debug3) << hash.size() << std::endl;

    log()->get(LogLevel::Debug3) << "Computing morphological opening\n";

    std::vector<double> minZ(np), maxZ(np);
    typedef std::vector<PointId> PointIdVec;
    std::map<PointId, PointIdVec> neighborMap;

    // instead of iterating over points, iterate over cells?
    // okay, this is almost the approx. version right?
    // any benefit?
    // what about hexes? or actual circular kernel?

    // erode
    for (PointId i = 0; i < np; ++i)
    {
        double x = view->getFieldAs<double>(Dimension::Id::X, i);
        double y = view->getFieldAs<double>(Dimension::Id::Y, i);

        // get hash for current point and neighboring cells, gather PointIds
        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int xIndex = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int yIndex = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);

        std::vector<PointId> ids;
        for (int row = yIndex-radius; row <= yIndex+radius; ++row)
        {
            if (row < 0 || row > (m_numRows-1))
                continue;
            for (int col = xIndex-radius; col <= xIndex+radius; ++col)
            {
                if (col < 0 || col > (m_numCols-1))
                    continue;
                int id = row * m_numCols + col;
                auto partial = hash[id];
                if (partial.size())
                {
                    ids.insert(ids.end(), partial.begin(), partial.end());
                }
            }
        }
        // log()->get(LogLevel::Debug3) << ids.size() << std::endl;

        neighborMap[i] = ids;
        double localMin(std::numeric_limits<double>::max());
        for (auto const& j : ids)
        {
            double z = view->getFieldAs<double>(Dimension::Id::Z, j);
            if (z < localMin)
                localMin = z;
        }
        minZ[i] = localMin;
    }

    // dilate
    for (PointId i = 0; i < np; ++i)
    {
        auto ids = neighborMap[i];
        double localMax(std::numeric_limits<double>::lowest());
        for (auto const& j : ids)
        {
            double z = minZ[j];
            if (z > localMax)
                localMax = z;
        }
        maxZ[i] = localMax;
    }

    return maxZ;
}

std::vector<double> MongusFilter::morphClose(PointViewPtr view, int radius)
{
    point_count_t np(view->size());

    auto hash = calculateHash(view);
    log()->get(LogLevel::Debug3) << hash.size() << std::endl;

    log()->get(LogLevel::Debug3) << "Computing morphological closing\n";

    std::vector<double> minZ(np), maxZ(np);
    typedef std::vector<PointId> PointIdVec;
    std::map<PointId, PointIdVec> neighborMap;

    // instead of iterating over points, iterate over cells?
    // okay, this is almost the approx. version right?
    // any benefit?
    // what about hexes? or actual circular kernel?

    // erode
    for (PointId i = 0; i < np; ++i)
    {
        double x = view->getFieldAs<double>(Dimension::Id::X, i);
        double y = view->getFieldAs<double>(Dimension::Id::Y, i);

        // get hash for current point and neighboring cells, gather PointIds
        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int xIndex = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int yIndex = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);

        std::vector<PointId> ids;
        for (int row = yIndex-radius; row <= yIndex+radius; ++row)
        {
            if (row < 0 || row > (m_numRows-1))
                continue;
            for (int col = xIndex-radius; col <= xIndex+radius; ++col)
            {
                if (col < 0 || col > (m_numCols-1))
                    continue;
                int id = row * m_numCols + col;
                auto partial = hash[id];
                if (partial.size())
                {
                    ids.insert(ids.end(), partial.begin(), partial.end());
                }
            }
        }
        // log()->get(LogLevel::Debug3) << ids.size() << std::endl;

        neighborMap[i] = ids;
        double localMax(std::numeric_limits<double>::lowest());
        for (auto const& j : ids)
        {
            double z = view->getFieldAs<double>(Dimension::Id::Z, j);
            if (z > localMax)
                localMax = z;
        }
        maxZ[i] = localMax;
    }

    // dilate
    for (PointId i = 0; i < np; ++i)
    {
        auto ids = neighborMap[i];
        double localMin(std::numeric_limits<double>::max());
        for (auto const& j : ids)
        {
            double z = maxZ[j];
            if (z < localMin)
                localMin = z;
        }
        minZ[i] = localMin;
    }

    return minZ;
}

void MongusFilter::TPS(PointViewPtr control)
{
    using namespace Eigen;

    point_count_t np(control->size());

    // determine size of H and S
    double cell_size = 16.0;
    int num_cols = static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/cell_size)) + 1;
    int num_rows = static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/cell_size)) + 1;
    int max_row = m_bounds.miny + num_rows * cell_size;

    // initialize H and S
    MatrixXd H = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::max());
    MatrixXd Xs = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::quiet_NaN());
    MatrixXd Ys = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::quiet_NaN());
    MatrixXd S = MatrixXd::Zero(num_rows, num_cols);

    auto clamp = [](double t, double min, double max)
    {
        return ((t < min) ? min : ((t > max) ? max : t));
    };

    // populate H with min Z values
    for (PointId i = 0; i < np; ++i)
    {
        using namespace Dimension::Id;
        double x = control->getFieldAs<double>(X, i);
        double y = control->getFieldAs<double>(Y, i);
        double z = control->getFieldAs<double>(Z, i);

        int xIndex = clamp(static_cast<int>(floor((x - m_bounds.minx) / cell_size)), 0, num_cols-1);
        int yIndex = clamp(static_cast<int>(floor((max_row - y) / cell_size)), 0, num_rows-1);

        if (z < H(yIndex, xIndex))
        {
            Xs(yIndex, xIndex) = x;
            Ys(yIndex, xIndex) = y;
            H(yIndex, xIndex) = z;
        }
    }

    for (int outer_row = 0; outer_row < num_rows; ++outer_row)
    {
        for (int outer_col = 0; outer_col < num_cols; ++outer_col)
        {
            int radius = 3;

            int col_start = clamp(outer_col-radius, 0, num_cols-1);
            int col_end = clamp(outer_col+radius, 0, num_cols-1);
            int col_size = col_end - col_start + 1;
            int row_start = clamp(outer_row-radius, 0, num_rows-1);
            int row_end = clamp(outer_row+radius, 0, num_rows-1);
            int row_size = row_end - row_start + 1;

            MatrixXd Xn = Xs.block(row_start, col_start, row_size, col_size);
            MatrixXd Yn = Ys.block(row_start, col_start, row_size, col_size);
            MatrixXd Hn = H.block(row_start, col_start, row_size, col_size);
            
            int nsize = Hn.size();
            VectorXd T = VectorXd::Zero(nsize);
            MatrixXd P = MatrixXd::Zero(nsize, 3);
            MatrixXd K = MatrixXd::Zero(nsize, nsize);

            for (int id = 0; id < Hn.size(); ++id)
            {
                double xj = Xn(id);
                if (std::isnan(xj))
                    continue;
                double yj = Yn(id);
                if (std::isnan(yj))
                    continue;
                double zj = Hn(id);
                if (zj == std::numeric_limits<double>::max())
                    continue;
                T(id) = zj;
                P.row(id) << 1, xj, yj;
                PointId kk = 0;
                for (int id2 = 0; id2 < Hn.size(); ++id2)
                {
                    if (id == id2)
                        continue;
                    double xk = Xn(id2);
                    if (std::isnan(xk))
                        continue;
                    double yk = Yn(id2);
                    if (std::isnan(yk))
                        continue;
                    double rsqr = (xj - xk) * (xj - xk) + (yj - yk) * (yj - yk);
                    if (rsqr == 0.0)
                        continue;
                    K(id, id2) = rsqr * std::log10(std::sqrt(rsqr));
                    kk++;
                }
            }

            MatrixXd A = MatrixXd::Zero(nsize+3, nsize+3);
            A.block(0,0,nsize,nsize) = K;
            A.block(0,nsize,nsize,3) = P;
            A.block(nsize,0,3,nsize) = P.transpose();

            VectorXd b = VectorXd::Zero(nsize+3);
            b.head(nsize) = T;

            VectorXd x = A.fullPivHouseholderQr().solve(b);

            Vector3d a = x.tail(3);
            VectorXd w = x.head(nsize);
            
            // std::cerr << "A:\n" << A << std::endl;
            // std::cerr << "x:\n" << x << std::endl;
            // std::cerr << "b:\n" << b << std::endl;

            double sum = 0.0;
            // double xsum = 0.0;
            // double ysum = 0.0;
            double xi = m_bounds.minx + outer_col * cell_size + cell_size / 2;
            double xi2 = Xs(outer_row, outer_col);
            // std::cerr << xi << "\t" << xi2 << std::endl;
            // if (std::isnan(xi))
                // continue;
            double yi = max_row - (outer_row * cell_size + cell_size / 2);
            double yi2 = Ys(outer_row, outer_col);
            // std::cerr << yi << "\t" << yi2 << std::endl;
            // if (std::isnan(yi))
                // continue;
            double zi = H(outer_row, outer_col);
            if (zi == std::numeric_limits<double>::max())
                continue;
                // std::cerr << nsize << "\t" << Xn.size() << std::endl;
            for (int j = 0; j < nsize; ++j)
            {
                double xj = Xn(j);
                if (std::isnan(xj))
                    continue;
                double yj = Yn(j);
                if (std::isnan(yj))
                    continue;
                // xsum += w(j) * xj;
                // ysum += w(j) * yj;
                // std::cerr << w(j) << "\t" << xj << "\t" << xsum << std::endl;
                double rsqr = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi);
                if (rsqr == 0.0)
                    continue;
                sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
            }
            
            // std::cerr << w.dot(P.col(1)) << std::endl;
            // std::cerr << w.dot(P.col(2)) << std::endl;
            
            // std::cerr << w.transpose() << std::endl;
            // std::cerr << P.col(1).transpose() << std::endl;
            // VectorXd foo(Map<VectorXd>(Xn.data(), Xn.cols()*Xn.rows()));
            // std::cerr << foo.transpose() << std::endl;
            // std::cerr << xsum << std::endl;
            // std::cerr << w.dot(P.col(1)) << std::endl;
            
            double val = a(0) + a(1)*xi + a(2)*yi + sum;
            
            std::cerr << std::fixed;
            std::cerr << std::setprecision(3)
                      << std::left
                      << "S(" << outer_row << "," << outer_col << "): "
                      << std::setw(10)
                      << val
                      << std::setw(3)
                      << "\tz: "
                      << std::setw(10)
                      << zi
                      << std::setw(7)
                      << "\tzdiff: "
                      << std::setw(5)
                      << zi - val
                      << std::setw(7)
                      << "\txdiff: "
                      << std::setw(5)
                      << xi2 - xi
                      << std::setw(7)
                      << "\tydiff: "
                      << std::setw(5)
                      << yi2 - yi
                      << std::setw(7)
                      << "\t# pts: "
                      << std::setw(3)
                      << nsize
                      << std::setw(5)
                      << "\tsum: "
                      << std::setw(10)
                      << sum
                      << std::setw(9)
                      << "\tw.sum(): "
                      << std::setw(5)
                      << w.sum()
                      << std::setw(6)
                      << "\txsum: "
                      << std::setw(5)
                      << w.dot(P.col(1))
                      << std::setw(6)
                      << "\tysum: "
                      << std::setw(5)
                      << w.dot(P.col(2))
                      << std::setw(3)
                      << "\ta: "
                      << std::setw(8)
                      << a.transpose()
                      << std::endl;
        }
    }
}

std::vector<double> MongusFilter::thinPlateSpline(PointViewPtr view, int radius)
{
    using namespace Eigen;

    double cellsize = 4.0;

    point_count_t np(view->size());

    auto hash = calculateHash(view, cellsize);
    log()->get(LogLevel::Debug3) << hash.size() << std::endl;

    log()->get(LogLevel::Debug3) << "Computing thin plate spline\n";

    std::cerr << hash.size() << std::endl;

    for (auto const& h : hash)
    {
        std::cerr << h.first << std::endl;

        double lowest = std::numeric_limits<double>::max();
        PointId i;
        std::cerr << "Number of indices for id " << h.first << ": " << h.second.size() << std::endl;
        for (auto const id : h.second)
        {
            double z = view->getFieldAs<double>(Dimension::Id::Z, id);
            if (z < lowest)
            {
                lowest = z;
                i = id;
            }
        }
        std::cerr << "Lowest is: " << i << ", " << lowest << std::endl;

        double xi = view->getFieldAs<double>(Dimension::Id::X, i);
        double yi = view->getFieldAs<double>(Dimension::Id::Y, i);

        // get hash for current point and neighboring cells, gather PointIds
        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int xIndex = clamp(static_cast<int>(floor((xi - m_bounds.minx) / cellsize)), 0, m_numCols-1);
        int yIndex = clamp(static_cast<int>(floor((m_maxRow - yi) / cellsize)), 0, m_numRows-1);

        std::vector<PointId> ids;
        for (int row = yIndex-radius; row <= yIndex+radius; ++row)
        {
            if (row < 0 || row > (m_numRows-1))
                continue;
            for (int col = xIndex-radius; col <= xIndex+radius; ++col)
            {
                if (col < 0 || col > (m_numCols-1))
                    continue;
                int id = row * m_numCols + col;
                auto partial = hash[id];
                if (partial.size())
                {
                    ids.insert(ids.end(), partial.begin(), partial.end());
                }
            }
        }

        // std::cerr << ids.size() << std::endl;
        VectorXd T(ids.size());
        MatrixXd P(ids.size(), 3);
        MatrixXd K(ids.size(), ids.size());
        // std::cerr << T.size() << std::endl;
        // std::cerr << P.size() << std::endl;
        // std::cerr << K.size() << std::endl;
        PointId jj = 0;
        for (auto it1 = ids.begin(), end = ids.end(); it1 != end; ++it1)
        {
            double xj = view->getFieldAs<double>(Dimension::Id::X, *it1);
            double yj = view->getFieldAs<double>(Dimension::Id::Y, *it1);
            double zj = view->getFieldAs<double>(Dimension::Id::Z, *it1);
            T(jj) = zj;
            P.row(jj) << 1, xj, yj;
            // std::cerr << T(jj) << std::endl;
            // std::cerr << P.row(jj) << std::endl;
            PointId kk = 0;
            // for (auto it2 = std::next(it1); it2 != end; ++it2)
            for (auto it2 = ids.begin(), end = ids.end(); it2 != end; ++it2)
            {
                if (*it1 == *it2)
                    continue;
                // std::cerr << *it2 << "\t" << view->size() << std::endl;
                double xk = view->getFieldAs<double>(Dimension::Id::X, *it2);
                double yk = view->getFieldAs<double>(Dimension::Id::Y, *it2);
                double r = std::sqrt((xj - xk) * (xj - xk) + (yj - yk) * (yj - yk));
                // std::cerr << r << "\t" << jj << "\t" << kk << std::endl;
                K(jj, kk) = r*r * std::log10(r);
                // std::cerr << K(jj, kk) << std::endl;
                kk++;
            }
            jj++;
        }

        MatrixXd A = MatrixXd::Zero(ids.size()+3, ids.size()+3);
        A.block(0,0,ids.size(),ids.size()) = K;
        A.block(0,ids.size(),ids.size(),3) = P;
        A.block(ids.size(),0,3,ids.size()) = P.transpose();

        VectorXd b = VectorXd::Zero(ids.size()+3);
        b.head(ids.size()) = T;

        VectorXd x = A.fullPivHouseholderQr().solve(b);
        // std::cerr << x.transpose() << std::endl;

        Vector3d a = x.tail(3);
        VectorXd w = x.head(ids.size());

        // std::cerr << a << std::endl;

        // std::cerr << w.sum() << std::endl;

        double sum = 0.0;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (i == ids[j])
                continue;

            double xj = view->getFieldAs<double>(Dimension::Id::X, ids[j]);
            double yj = view->getFieldAs<double>(Dimension::Id::Y, ids[j]);
            double r = std::sqrt((xj - xi) * (xj - xi) + (yj - yi) * (yj - yi));
            sum += w(j) * r * r * std::log10(r);
        }

        std::cerr << ids.size() << "\t" << a(0) + a(1)*xi + a(2)*yi + sum << std::endl;

        // S(x, y) = a0 + (a1*x) + (a2*y) + /sum (w_i * R(|| (x_i, y_i) - (x, y)||)
    }

    // return minZ;
}

PointIdHash MongusFilter::calculateHash(PointViewPtr view)
{
    log()->get(LogLevel::Debug3) << "Recomputing hash\n";

    // compute hash grid at m_cellSize
    PointIdHash hash;
    hash.reserve(m_numCols * m_numRows);

    for (PointId i = 0; i < view->size(); ++i)
    {
        double x = view->getFieldAs<double>(Dimension::Id::X, i);
        double y = view->getFieldAs<double>(Dimension::Id::Y, i);

        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int xIndex = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int yIndex = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);
        // log()->get(LogLevel::Debug5) << xIndex << ", " << yIndex << std::endl;

        int id = yIndex * m_numCols + xIndex;

        auto ids = hash[id];
        ids.push_back(i);
        hash[id] = ids;
    }

    return hash;
}

PointIdHash MongusFilter::calculateHash(PointViewPtr view, double cell_size)
{
    log()->get(LogLevel::Debug3) << "Recomputing hash\n";

    int num_cols = static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/cell_size)) + 1;
    int num_rows = static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/cell_size)) + 1;
    int max_row = m_bounds.miny + num_rows * cell_size;

    // compute hash grid at cell_size
    PointIdHash hash;
    hash.reserve(num_cols * num_rows);

    for (PointId i = 0; i < view->size(); ++i)
    {
        double x = view->getFieldAs<double>(Dimension::Id::X, i);
        double y = view->getFieldAs<double>(Dimension::Id::Y, i);

        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int xIndex = clamp(static_cast<int>(floor((x - m_bounds.minx) / cell_size)), 0, num_cols-1);
        int yIndex = clamp(static_cast<int>(floor((max_row - y) / cell_size)), 0, num_rows-1);
        // log()->get(LogLevel::Debug5) << xIndex << ", " << yIndex << std::endl;

        int id = yIndex * num_cols + xIndex;

        auto ids = hash[id];
        ids.push_back(i);
        hash[id] = ids;
    }
    std::cerr << num_cols << "\t" << num_rows << std::endl;
    std::cerr << hash.size() << std::endl;
    int s = 0;
    for (auto const& h : hash)
    {
        s += h.second.size();
        // if (h.second.size() == 0)
        std::cerr << h.first << std::endl;
    }
    std::cerr << s << std::endl;

    return hash;
}

std::vector<PointId> MongusFilter::processGround(PointViewPtr view)
{
    point_count_t np(view->size());

    std::vector<PointId> groundIdx;
    for (PointId i = 0; i < np; ++i)
        groundIdx.push_back(i);

    // Create new cloud to hold the filtered results. Apply the morphological
    // opening operation at the current window size.
    // auto maxZ = morphOpen(ground, window_sizes[j]*0.5);
}

PointViewSet MongusFilter::run(PointViewPtr input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process MongusFilter...\n";

    // auto idx = processGround(input);

    PointViewSet viewSet;

    // initialization

    // start by stashing the original Z values
    std::vector<double> originalZ;
    for (PointId i = 0; i < input->size(); ++i)
    {
        originalZ.push_back(input->getFieldAs<double>(Dimension::Id::Z, i));
    }

    input->calculateBounds(m_bounds);

    m_cellSize = 1.0;

    m_numCols = static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/m_cellSize)) + 1;
    m_numRows = static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/m_cellSize)) + 1;
    m_maxRow = m_bounds.miny + m_numRows * m_cellSize;

    auto oz = morphOpen(input, 11);

    // Z now contains the opened values
    for (PointId i = 0; i < input->size(); ++i)
    {
        input->setField(Dimension::Id::Z, i, oz[i]);
    }

    auto cz = morphClose(input, 9);

    // Z now contains the closed values
    for (PointId i = 0; i < input->size(); ++i)
    {
        input->setField(Dimension::Id::Z, i, cz[i]);
    }

    // compare morphological values to orignals and replace low points
    for (PointId i = 0; i < input->size(); ++i)
    {
        auto diff = input->getFieldAs<double>(Dimension::Id::Z, i) - originalZ[i];

        if (diff >= 1.0)
        {
            // std::cerr << diff << std::endl;
            continue;
        }
        else
        {
            input->setField(Dimension::Id::Z, i, originalZ[i]);
        }
    }

    // control point selection
    // start at m_cellSize?, then downsample by 2 with each successive level
    // lowest points from control points

    // TPS
    // in practice a 7 x 7 neighborhood is used
    /*
     Matrix3f A; // A is [ K P; P' 0 ]
                 // where Kij = R(||(x_i, y_i) - (x_j, y_j)||),
                 // and R(r) = r^2 log r
                 // Pi = (1, x_i, y_i)
     Vector3f b; // b is [ T; 0 ]
                 // where T is a tensor of control points
                 // (must be the z_i values?)
     Vector3f x = A.colPivHouseholderQr().solve(b);
                 // x is [ w; a ]
     // finally
     // S(x, y) = a0 + (a1*x) + (a2*y) + /sum (w_i * R(|| (x_i, y_i) - (x, y)||)
     */
    TPS(input);

    viewSet.insert(input);

    // auto cz = morphClose(input, 9);


    // if (!idx.empty() && (m_classify || m_extract))
    // {
    //
    //     if (m_classify)
    //     {
    //         log()->get(LogLevel::Debug2) << "Labeled " << idx.size() << " ground returns!\n";
    //
    //         // set the classification label of ground returns as 2
    //         // (corresponding to ASPRS LAS specification)
    //         for (const auto& i : idx)
    //         {
    //             input->setField(Dimension::Id::Classification, i, 2);
    //         }
    //
    //         viewSet.insert(input);
    //     }
    //
    //     if (m_extract)
    //     {
    //         log()->get(LogLevel::Debug2) << "Extracted " << idx.size() << " ground returns!\n";
    //
    //         // create new PointView containing only ground returns
    //         PointViewPtr output = input->makeNew();
    //         for (const auto& i : idx)
    //         {
    //             output->appendPoint(*input, i);
    //         }
    //
    //         viewSet.erase(input);
    //         viewSet.insert(output);
    //     }
    // }
    // else
    // {
    //     if (idx.empty())
    //         log()->get(LogLevel::Debug2) << "Filtered cloud has no ground returns!\n";
    //
    //     if (!(m_classify || m_extract))
    //         log()->get(LogLevel::Debug2) << "Must choose --classify or --extract\n";
    //
    //     // return the input buffer unchanged
    //     viewSet.insert(input);
    // }

    return viewSet;
}

} // namespace pdal
