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

#include "MongusFilter.hpp"

#include <pdal/pdal_macros.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Dense>

#include <unordered_map>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.mongus", "Mongus and Zalik (2012)",
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

int MongusFilter::clamp(int t, int min, int max)
{
    return ((t < min) ? min : ((t > max) ? max : t));
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

        int c = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int r = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);

        std::vector<PointId> ids;

        int col_start = clamp(c-radius, 0, m_numCols-1);
        int col_end = clamp(c+radius, 0, m_numCols-1);
        int row_start = clamp(r-radius, 0, m_numRows-1);
        int row_end = clamp(r+radius, 0, m_numRows-1);

        for (int row = row_start; row <= row_end; ++row)
        {
            for (int col = col_start; col <= col_end; ++col)
            {
                int id = row * m_numCols + col;
                auto partial = hash[id];
                if (partial.size())
                {
                    ids.insert(ids.end(), partial.begin(), partial.end());
                }
            }
        }

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

        int c = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int r = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);

        std::vector<PointId> ids;

        int col_start = clamp(c-radius, 0, m_numCols-1);
        int col_end = clamp(c+radius, 0, m_numCols-1);
        int row_start = clamp(r-radius, 0, m_numRows-1);
        int row_end = clamp(r+radius, 0, m_numRows-1);

        for (int row = row_start; row <= row_end; ++row)
        {
            for (int col = col_start; col <= col_end; ++col)
            {
                int id = row * m_numCols + col;
                auto partial = hash[id];
                if (partial.size())
                {
                    ids.insert(ids.end(), partial.begin(), partial.end());
                }
            }
        }

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

Eigen::MatrixXd MongusFilter::TPS(Eigen::MatrixXd cx, Eigen::MatrixXd cy, Eigen::MatrixXd cz, double cell_size)
{
    using namespace Eigen;

    int num_rows = cz.rows();
    int num_cols = cz.cols();
    int max_row = m_bounds.miny + num_rows * cell_size;

    MatrixXd S = MatrixXd::Zero(num_rows, num_cols);

    for (int outer_row = 0; outer_row < num_rows; ++outer_row)
    {
        for (int outer_col = 0; outer_col < num_cols; ++outer_col)
        {
            // radius of 3, giving an effective 7x7 neighborhood, according to
            // the paper
            int radius = 3;

            int col_start = clamp(outer_col-radius, 0, num_cols-1);
            int col_end = clamp(outer_col+radius, 0, num_cols-1);
            int col_size = col_end - col_start + 1;
            int row_start = clamp(outer_row-radius, 0, num_rows-1);
            int row_end = clamp(outer_row+radius, 0, num_rows-1);
            int row_size = row_end - row_start + 1;

            MatrixXd Xn = cx.block(row_start, col_start, row_size, col_size);
            MatrixXd Yn = cy.block(row_start, col_start, row_size, col_size);
            MatrixXd Hn = cz.block(row_start, col_start, row_size, col_size);

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

            double sum = 0.0;
            double xi = m_bounds.minx + outer_col * cell_size + cell_size / 2;
            double xi2 = cx(outer_row, outer_col);
            double yi = max_row - (outer_row * cell_size + cell_size / 2);
            double yi2 = cy(outer_row, outer_col);
            double zi = cz(outer_row, outer_col);
            if (zi == std::numeric_limits<double>::max())
                continue;
            for (int j = 0; j < nsize; ++j)
            {
                double xj = Xn(j);
                if (std::isnan(xj))
                    continue;
                double yj = Yn(j);
                if (std::isnan(yj))
                    continue;
                double rsqr = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi);
                if (rsqr == 0.0)
                    continue;
                sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
            }

            S(outer_row, outer_col) = a(0) + a(1)*xi + a(2)*yi + sum;

            // std::cerr << std::fixed;
            // std::cerr << std::setprecision(3)
            //           << std::left
            //           << "S(" << outer_row << "," << outer_col << "): "
            //           << std::setw(10)
            //           << S(outer_row, outer_col)
            //           << std::setw(3)
            //           << "\tz: "
            //           << std::setw(10)
            //           << zi
            //           << std::setw(7)
            //           << "\tzdiff: "
            //           << std::setw(5)
            //           << zi - S(outer_row, outer_col)
            //           << std::setw(7)
            //           << "\txdiff: "
            //           << std::setw(5)
            //           << xi2 - xi
            //           << std::setw(7)
            //           << "\tydiff: "
            //           << std::setw(5)
            //           << yi2 - yi
            //           << std::setw(7)
            //           << "\t# pts: "
            //           << std::setw(3)
            //           << nsize
            //           << std::setw(5)
            //           << "\tsum: "
            //           << std::setw(10)
            //           << sum
            //           << std::setw(9)
            //           << "\tw.sum(): "
            //           << std::setw(5)
            //           << w.sum()
            //           << std::setw(6)
            //           << "\txsum: "
            //           << std::setw(5)
            //           << w.dot(P.col(1))
            //           << std::setw(6)
            //           << "\tysum: "
            //           << std::setw(5)
            //           << w.dot(P.col(2))
            //           << std::setw(3)
            //           << "\ta: "
            //           << std::setw(8)
            //           << a.transpose()
            //           << std::endl;
        }
    }

    return S;
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

        int c = clamp(static_cast<int>(floor((x - m_bounds.minx) / m_cellSize)), 0, m_numCols-1);
        int r = clamp(static_cast<int>(floor((m_maxRow - y) / m_cellSize)), 0, m_numRows-1);

        int id = r * m_numCols + c;

        auto ids = hash[id];
        ids.push_back(i);
        hash[id] = ids;
    }

    return hash;
}

std::vector<PointId> MongusFilter::processGround(PointViewPtr view)
{
    point_count_t np(view->size());

    std::vector<PointId> groundIdx;

    // initialization

    // start by stashing the original Z values
    std::vector<double> originalZ;
    for (PointId i = 0; i < view->size(); ++i)
    {
        originalZ.push_back(view->getFieldAs<double>(Dimension::Id::Z, i));
    }

    view->calculateBounds(m_bounds);

    m_cellSize = 1.0;

    m_numCols = static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/m_cellSize)) + 1;
    m_numRows = static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/m_cellSize)) + 1;
    m_maxRow = m_bounds.miny + m_numRows * m_cellSize;

    auto mo = morphOpen(view, 11);

    // Z now contains the opened values
    for (PointId i = 0; i < view->size(); ++i)
    {
        view->setField(Dimension::Id::Z, i, mo[i]);
    }

    auto mc = morphClose(view, 9);

    // Z now contains the closed values
    for (PointId i = 0; i < view->size(); ++i)
    {
        view->setField(Dimension::Id::Z, i, mc[i]);
    }

    // compare morphological values to orignals and replace low points
    for (PointId i = 0; i < view->size(); ++i)
    {
        auto diff = view->getFieldAs<double>(Dimension::Id::Z, i) - originalZ[i];

        if (diff < 1.0)
            view->setField(Dimension::Id::Z, i, originalZ[i]);
    }

    // create control points matrix at 1m cell size
    double initCellSize = 1.0;

    int num_cols = static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/initCellSize)) + 1;
    int num_rows = static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/initCellSize)) + 1;
    int max_row = m_bounds.miny + num_rows * initCellSize;

    using namespace Eigen;

    MatrixXd cx = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::quiet_NaN());
    MatrixXd cy = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::quiet_NaN());
    MatrixXd cz = MatrixXd::Constant(num_rows, num_cols, std::numeric_limits<double>::max());

    for (PointId i = 0; i < view->size(); ++i)
    {
        using namespace Dimension::Id;
        double x = view->getFieldAs<double>(X, i);
        double y = view->getFieldAs<double>(Y, i);
        double z = view->getFieldAs<double>(Z, i);

        int c =
            clamp(static_cast<int>(floor((x - m_bounds.minx) / initCellSize)),
                  0, num_cols-1);
        int r =
            clamp(static_cast<int>(floor((max_row - y) / initCellSize)),
                  0, num_rows-1);

        if (z < cz(r, c))
        {
            cx(r, c) = x;
            cy(r, c) = y;
            cz(r, c) = z;
        }
    }

    // downsample control at max_level
    int level = 8;
    double newCellSize = std::pow(2, level-1);

    MatrixXd dcx, dcy, dcz;
    downsampleMin(&cx, &cy, &cz, &dcx, &dcy, &dcz, newCellSize);

    // compute TPS at max_level
    auto surface = TPS(dcx, dcy, dcz, newCellSize);

    int end_level = 4;

    MatrixXd t;

    // for each level counting back to 0
    for (int l = level-1; l >= end_level; --l)
    {
        std::cerr << "Level " << l << std::endl;

        // downsample control at level
        newCellSize = std::pow(2, l-1);
        downsampleMin(&cx, &cy, &cz, &dcx, &dcy, &dcz, newCellSize);

        // applyTopHat(&dcz, &surface, 2*l);
        auto R = computeResidual(dcz, surface);
        std::cerr << R << std::endl << std::endl;
        auto maxZ = matrixOpen(R, 2*l);
        std::cerr << maxZ << std::endl << std::endl;
        MatrixXd T = R - maxZ;
        std::cerr << T << std::endl << std::endl;
        t = computeThresholds(T, 2*l);
        std::cerr << t << std::endl << std::endl;
        for (int r = 0; r < T.rows(); ++r)
        {
            for (int c = 0; c < T.cols(); ++c)
            {
                if (T(r,c) > t(r,c))
                {
                    std::cerr << T(r, c) << " > " << t(r, c) << " at (" << r << "," << c << ")" << std::endl;
                    dcz(r, c) = std::numeric_limits<double>::max();
                }
            }
        }

        // compute TPS with update control at level
        surface.resize(dcz.rows(), dcz.cols());
        surface = TPS(dcx, dcy, dcz, newCellSize);
    }
    std::cerr << "check\n";

    // apply final filtering (top hat) using raw points against TPS
    for (PointId i = 0; i < view->size(); ++i)
    {
        using namespace Dimension::Id;
        double x = view->getFieldAs<double>(X, i);
        double y = view->getFieldAs<double>(Y, i);
        double z = view->getFieldAs<double>(Z, i);

        int c =
            clamp(static_cast<int>(floor((x - m_bounds.minx) / newCellSize)),
                  0, surface.rows()-1);
        int r =
            clamp(static_cast<int>(floor((max_row - y) / newCellSize)),
                  0, surface.cols()-1);

        double res = z - surface(r, c);
        // std::cerr << z << "\t" << surface(r, c) << "\t" << res << "\t" << t(r, c) << std::endl;
        // open?
        if (res > t(r, c))
            continue;
        groundIdx.push_back(i);
    }

    return groundIdx;

    // temporary
    // PointViewPtr output = view->makeNew();
    //
    // PointId i = 0;
    // for (int r = 0; r < surface.rows(); ++r)
    // {
    //     for (int c = 0; c < surface.cols(); ++c)
    //     {
    //         using namespace Dimension::Id;
    //         if (std::isnan(dcx(r, c)))
    //             continue;
    //         output->setField(X, i, dcx(r, c));
    //         if (std::isnan(dcy(r, c)))
    //             continue;
    //         output->setField(Y, i, dcy(r, c));
    //         if (surface(r, c) == 0)
    //             continue;
    //         output->setField(Z, i, surface(r, c));
    //         i++;
    //     }
    // }
    //
    // viewSet.erase(view);
    // viewSet.insert(output);
}

void MongusFilter::downsampleMin(Eigen::MatrixXd *cx, Eigen::MatrixXd *cy, Eigen::MatrixXd* cz, Eigen::MatrixXd *dcx, Eigen::MatrixXd *dcy, Eigen::MatrixXd* dcz, double cell_size)
{
    int nr = ceil(cz->rows() / cell_size);
    int nc = ceil(cz->cols() / cell_size);

    std::cerr << nr << "\t" << nc << std::endl;

    dcx->resize(nr, nc);
    dcx->setConstant(std::numeric_limits<double>::quiet_NaN());
    dcy->resize(nr, nc);
    dcy->setConstant(std::numeric_limits<double>::quiet_NaN());
    dcz->resize(nr, nc);
    dcz->setConstant(std::numeric_limits<double>::max());

    for (int r = 0; r < cz->rows(); ++r)
    {
        for (int c = 0; c < cz->cols(); ++c)
        {
            if ((*cz)(r, c) == std::numeric_limits<double>::max())
                continue;

            int rr = std::floor(r/cell_size);
            int cc = std::floor(c/cell_size);

            if ((*cz)(r, c) < (*dcz)(rr, cc))
            {
                (*dcx)(rr, cc) = (*cx)(r, c);
                (*dcy)(rr, cc) = (*cy)(r, c);
                (*dcz)(rr, cc) = (*cz)(r, c);
            }
        }
    }
}

Eigen::MatrixXd MongusFilter::computeResidual(Eigen::MatrixXd cz, Eigen::MatrixXd surface)
{
    using namespace Eigen;

    MatrixXd R = MatrixXd::Zero(cz.rows(), cz.cols());
    for (int r = 0; r < cz.rows(); ++r)
    {
        for (int c = 0; c < cz.cols(); ++c)
        {
            if (cz(r, c) == std::numeric_limits<double>::max())
                continue;

            int rr = std::floor(r/2);
            int cc = std::floor(c/2);
            R(r, c) = cz(r, c) - surface(rr, cc);
        }
    }

    return R;
}

Eigen::MatrixXd MongusFilter::matrixOpen(Eigen::MatrixXd data, int radius)
{
    using namespace Eigen;

    int new_num_rows = data.rows();
    int new_num_cols = data.cols();

    // first min, then max of min
    MatrixXd minZ =
        MatrixXd::Constant(new_num_rows, new_num_cols,
                           std::numeric_limits<double>::max());
    MatrixXd maxZ =
        MatrixXd::Constant(new_num_rows, new_num_cols,
                           std::numeric_limits<double>::lowest());
    for (int r = 0; r < minZ.rows(); ++r)
    {
        for (int c = 0; c < minZ.cols(); ++c)
        {
            int col_start = clamp(c-radius, 0, new_num_cols-1);
            int col_end = clamp(c+radius, 0, new_num_cols-1);
            int row_start = clamp(r-radius, 0, new_num_rows-1);
            int row_end = clamp(r+radius, 0, new_num_rows-1);

            for (int row = row_start; row <= row_end; ++row)
            {
                for (int col = col_start; col <= col_end; ++col)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (data(row, col) < minZ(r, c))
                        minZ(r, c) = data(row, col);
                }
            }
        }
    }
    for (int r = 0; r < minZ.rows(); ++r)
    {
        for (int c = 0; c < minZ.cols(); ++c)
        {
            for (int row = r-radius; row <= r+radius; ++row)
            {
                int col_start = clamp(c-radius, 0, new_num_cols-1);
                int col_end = clamp(c+radius, 0, new_num_cols-1);
                int row_start = clamp(r-radius, 0, new_num_rows-1);
                int row_end = clamp(r+radius, 0, new_num_rows-1);

                for (int row = row_start; row <= row_end; ++row)
                {
                    for (int col = col_start; col <= col_end; ++col)
                    {
                        if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                            continue;
                        if (minZ(row, col) > maxZ(r, c))
                            maxZ(r, c) = minZ(row, col);
                    }
                }
            }
        }

        return maxZ;
    }

    Eigen::MatrixXd MongusFilter::computeThresholds(Eigen::MatrixXd T, int radius)
    {
        using namespace Eigen;

        int new_num_rows = T.rows();
        int new_num_cols = T.cols();

        MatrixXd t = MatrixXd::Zero(new_num_rows, new_num_cols);
        for (int r = 0; r < T.rows(); ++r)
        {
            for (int c = 0; c < T.cols(); ++c)
            {
                double M1 = 0;
                double M2 = 0;
                int n = 0;

                int col_start = clamp(c-radius, 0, new_num_cols-1);
                int col_end = clamp(c+radius, 0, new_num_cols-1);
                int col_size = col_end - col_start + 1;
                int row_start = clamp(r-radius, 0, new_num_rows-1);
                int row_end = clamp(r+radius, 0, new_num_rows-1);
                int row_size = row_end - row_start + 1;

                for (int row = row_start; row <= row_end; ++row)
                {
                    for (int col = col_start; col <= col_end; ++col)
                    {
                        int n1 = n;
                        n++;
                        double delta = T(row, col) - M1;
                        double delta_n = delta / n;
                        double term1 = delta * delta_n * n1;
                        M1 += delta_n;
                        M2 += term1;
                    }
                }
                // std::cerr << M1 << "\t" << std::sqrt(M2/(n-1)) << "\t" << n << std::endl;
                t(r, c) = M1 + 3 * std::sqrt(M2/(n-1));
            }
        }

        return t;
    }

    void MongusFilter::applyTopHat(Eigen::MatrixXd *cz, Eigen::MatrixXd *surface, int radius)
    {
        using namespace Eigen;

        auto R = computeResidual(*cz, *surface);
        auto maxZ = matrixOpen(R, radius);
        MatrixXd T = R - maxZ;
        auto t = computeThresholds(T, radius);
        for (int r = 0; r < T.rows(); ++r)
        {
            for (int c = 0; c < T.cols(); ++c)
            {
                if (T(r,c) > t(r,c))
                {
                    std::cerr << T(r, c) << " > " << t(r, c) << " at (" << r << "," << c << ")" << std::endl;
                    (*cz)(r, c) = std::numeric_limits<double>::max();
                }
            }
        }
    }

    PointViewSet MongusFilter::run(PointViewPtr view)
    {
        bool logOutput = log()->getLevel() > LogLevel::Debug1;
        if (logOutput)
            log()->floatPrecision(8);
        log()->get(LogLevel::Debug2) << "Process MongusFilter...\n";

        auto idx = processGround(view);

        PointViewSet viewSet;

        if (!idx.empty() && (m_classify || m_extract))
        {

            if (m_classify)
            {
                log()->get(LogLevel::Debug2) << "Labeled " << idx.size() << " ground returns!\n";

                // set the classification label of ground returns as 2
                // (corresponding to ASPRS LAS specification)
                for (const auto& i : idx)
                {
                    view->setField(Dimension::Id::Classification, i, 2);
                }

                viewSet.insert(view);
            }

            if (m_extract)
            {
                log()->get(LogLevel::Debug2) << "Extracted " << idx.size() << " ground returns!\n";

                // create new PointView containing only ground returns
                PointViewPtr output = view->makeNew();
                for (const auto& i : idx)
                {
                    output->appendPoint(*view, i);
                }

                viewSet.erase(view);
                viewSet.insert(output);
            }
        }
        else
        {
            if (idx.empty())
                log()->get(LogLevel::Debug2) << "Filtered cloud has no ground returns!\n";

            if (!(m_classify || m_extract))
                log()->get(LogLevel::Debug2) << "Must choose --classify or --extract\n";

            // return the view buffer unchanged
            viewSet.insert(view);
        }

        return viewSet;
    }

} // namespace pdal
