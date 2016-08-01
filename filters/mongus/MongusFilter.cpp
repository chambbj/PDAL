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
#include <pdal/PipelineManager.hpp>
#include <buffer/BufferReader.hpp>
#include <pdal/util/ProgramArgs.hpp>

#include <Eigen/Dense>

#include "gdal_priv.h" // For File I/O
#include "gdal_version.h" // For version info
#include "ogr_spatialref.h"  //For Geographic Information/Transformations

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
    args.add("cell", "Cell size", m_cellSize, 1.0);
    args.add("k", "Stdev multiplier for threshold", m_k, 3.0);
    args.add("l", "Max level", m_l, 8);
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

int MongusFilter::getColIndex(double x, double cell_size)
{
    return static_cast<int>(floor((x - m_bounds.minx) / cell_size));
}

int MongusFilter::getRowIndex(double y, double cell_size)
{
    return static_cast<int>(floor((m_maxRow - y) / cell_size));
}

Eigen::MatrixXd MongusFilter::TPS(Eigen::MatrixXd cx, Eigen::MatrixXd cy,
                                  Eigen::MatrixXd cz, double cell_size)
{
    using namespace Eigen;

    int num_rows = cz.rows();
    int num_cols = cz.cols();
    int max_row = m_bounds.miny + num_rows * cell_size;

    MatrixXd S = MatrixXd::Zero(num_rows, num_cols);

    for (auto outer_col = 0; outer_col < num_cols; ++outer_col)
    {
        for (auto outer_row = 0; outer_row < num_rows; ++outer_row)
        {
            // Further optimizations are achieved by estimating only the
            // interpolated surface within a local neighbourhood (e.g. a 7 x 7
            // neighbourhood is used in our case) of the cell being filtered.
            int radius = 3;

            int cs = clamp(outer_col-radius, 0, num_cols-1);
            int ce = clamp(outer_col+radius, 0, num_cols-1);
            int col_size = ce - cs + 1;
            int rs = clamp(outer_row-radius, 0, num_rows-1);
            int re = clamp(outer_row+radius, 0, num_rows-1);
            int row_size = re - rs + 1;

            MatrixXd Xn = cx.block(rs, cs, row_size, col_size);
            MatrixXd Yn = cy.block(rs, cs, row_size, col_size);
            MatrixXd Hn = cz.block(rs, cs, row_size, col_size);

            int nsize = Hn.size();
            VectorXd T = VectorXd::Zero(nsize);
            MatrixXd P = MatrixXd::Zero(nsize, 3);
            MatrixXd K = MatrixXd::Zero(nsize, nsize);

            for (auto id = 0; id < Hn.size(); ++id)
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
                for (auto id2 = 0; id2 < Hn.size(); ++id2)
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
            for (auto j = 0; j < nsize; ++j)
            {
                double xj = Xn(j);
                if (std::isnan(xj))
                    continue;
                double yj = Yn(j);
                if (std::isnan(yj))
                    continue;
                double rsqr = (xj - xi2) * (xj - xi2) + (yj - yi2) * (yj - yi2);
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

void MongusFilter::writeMatrix(Eigen::MatrixXd data, std::string filename, double cell_size, PointViewPtr view)
{
    int cols = data.cols();
    int rows = data.rows();

    GDALAllRegister();

    GDALDataset *mpDstDS;

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

            mpDstDS = tpDriver->Create(filename.c_str(), cols, rows, 1,
                                       GDT_Float32, papszOptions);

            // set the geo transformation
            double adfGeoTransform[6];
            adfGeoTransform[0] = m_bounds.minx; // - 0.5*m_GRID_DIST_X;
            adfGeoTransform[1] = cell_size;
            adfGeoTransform[2] = 0.0;
            adfGeoTransform[3] = m_bounds.maxy; // + 0.5*m_GRID_DIST_Y;
            adfGeoTransform[4] = 0.0;
            adfGeoTransform[5] = -1 * cell_size;
            mpDstDS->SetGeoTransform(adfGeoTransform);

            // set the projection
            mpDstDS->SetProjection(view->spatialReference().getWKT().c_str());
        }
    }

    // if we have a valid file
    if (mpDstDS)
    {
        // loop over the raster and determine max slope at each location
        int cs = 1, ce = cols - 1;
        int rs = 1, re = rows - 1;
        float *poRasterData = new float[cols*rows];
        for (auto i=0; i<cols*rows; i++)
        {
            poRasterData[i] = std::numeric_limits<float>::min();
        }

        #pragma omp parallel for
        for (auto c = cs; c < ce; ++c)
        {
            for (auto r = rs; r < re; ++r)
            {
                if (data(r, c) == 0.0 || std::isnan(data(r, c) || data(r, c) == std::numeric_limits<double>::max()))
                    continue;
                poRasterData[(r * cols) + c] =
                    data(r, c);
            }
        }

        // write the data
        if (poRasterData)
        {
            GDALRasterBand *tBand = mpDstDS->GetRasterBand(1);

            tBand->SetNoDataValue(std::numeric_limits<float>::min());

            if (cols > 0 && rows > 0)
#if GDAL_VERSION_MAJOR <= 1
                tBand->RasterIO(GF_Write, 0, 0, cols, rows,
                                poRasterData, cols, rows,
                                GDT_Float32, 0, 0);
#else

                int ret = tBand->RasterIO(GF_Write, 0, 0, cols, rows,
                                          poRasterData, cols, rows,
                                          GDT_Float32, 0, 0, 0);
#endif
        }

        GDALClose((GDALDatasetH) mpDstDS);

        delete [] poRasterData;
    }
}

void MongusFilter::writeControl(Eigen::MatrixXd cx, Eigen::MatrixXd cy, Eigen::MatrixXd cz, std::string filename)
{
    using namespace Dimension;

    PipelineManager m;

    PointTable table;
    PointViewPtr view(new PointView(table));

    table.layout()->registerDim(Id::X);
    table.layout()->registerDim(Id::Y);
    table.layout()->registerDim(Id::Z);

    PointId i = 0;
    for (auto j = 0; j < cz.size(); ++j)
    {
        if (std::isnan(cx(j)) || std::isnan(cy(j)))
            continue;
        if (cz(j) == std::numeric_limits<double>::max())
            continue;
        view->setField(Id::X, i, cx(j));
        view->setField(Id::Y, i, cy(j));
        view->setField(Id::Z, i, cz(j));
        i++;
    }

    BufferReader r;
    r.addView(view);

    Stage& w = m.makeWriter(filename, "writers.las", r);
    w.prepare(table);
    w.execute(table);
}

std::vector<PointId> MongusFilter::processGround(PointViewPtr view)
{
    using namespace Eigen;

    point_count_t np(view->size());

    std::vector<PointId> groundIdx;

    // initialization

    view->calculateBounds(m_bounds);

    m_numCols =
        static_cast<int>(ceil((m_bounds.maxx - m_bounds.minx)/m_cellSize)) + 1;
    m_numRows =
        static_cast<int>(ceil((m_bounds.maxy - m_bounds.miny)/m_cellSize)) + 1;
    m_maxRow = m_bounds.miny + m_numRows * m_cellSize;

    // create control points matrix at default cell size
    MatrixXd cx(m_numRows, m_numCols);
    cx.setConstant(std::numeric_limits<double>::quiet_NaN());

    MatrixXd cy(m_numRows, m_numCols);
    cy.setConstant(std::numeric_limits<double>::quiet_NaN());

    MatrixXd cz(m_numRows, m_numCols);
    cz.setConstant(std::numeric_limits<double>::max());

    for (auto i = 0; i < np; ++i)
    {
        using namespace Dimension;
        double x = view->getFieldAs<double>(Id::X, i);
        double y = view->getFieldAs<double>(Id::Y, i);
        double z = view->getFieldAs<double>(Id::Z, i);

        int c = clamp(getColIndex(x, m_cellSize), 0, m_numCols-1);
        int r = clamp(getRowIndex(y, m_cellSize), 0, m_numRows-1);

        if (z < cz(r, c))
        {
            cx(r, c) = x;
            cy(r, c) = y;
            cz(r, c) = z;
        }
    }

    // In our case, 2D structural elements of circular shape are employed and
    // sufficient accuracy is achieved by using a larger window size for opening
    // (W11) than for closing (W9).
    MatrixXd mo = matrixOpen(cz, 11);
    MatrixXd mc = matrixClose(mo, 9);

    // ...in order to minimize the distortions caused by such filtering, the
    // output points ... are compared to C and only ci with significantly lower
    // elevation [are] replaced... In our case, d = 1.0 m was used.
    for (auto i = 0; i < cz.size(); ++i)
    {
        if ((mc(i) - cz(i)) >= 1.0)
            cz(i) = mc(i);
    }

    // downsample control at max_level
    int level = m_l;
    double cur_cell_size = m_cellSize * std::pow(2, level-1);

    MatrixXd dcx, dcy, dcz;

    // Top-level control samples are assumed to be ground points, no filtering
    // is applied.
    downsampleMin(&cx, &cy, &cz, &dcx, &dcy, &dcz, cur_cell_size);

    // Point-filtering is performed iteratively at each level of the
    // control-points hierarchy in a top-down fashion
    for (auto l = level-1; l > 0; --l)
    {
        std::cerr << "Level " << l << std::endl;

        // compute TPS with update control at level

        // The interpolated surface is estimated based on the filtered set of
        // TPS control-points at the previous level of hierarchy
        MatrixXd surface = TPS(dcx, dcy, dcz, cur_cell_size);

        // downsample control at level
        cur_cell_size /= 2;
        downsampleMin(&cx, &cy, &cz, &dcx, &dcy, &dcz, cur_cell_size);

        MatrixXd R = computeResidual(dcz, surface);
        MatrixXd maxZ = matrixOpen(R, 2*l);
        MatrixXd T = R - maxZ;
        MatrixXd t = computeThresholds(T, 2*l);

        // the time complexity of the approach is reduced by filtering only the
        // control-points in each iteration
        for (auto c = 0; c < T.cols(); ++c)
        {
            for (auto r = 0; r < T.rows(); ++r)
            {
                // If the TPS control-point is recognized as a non-ground point,
                // it is replaced by the interpolated point.
                if (T(r,c) > t(r,c))
                {
                    int rr = std::floor(r/2);
                    int cc = std::floor(c/2);
                    int rs = cur_cell_size * r;
                    int cs = cur_cell_size * c;
                    MatrixXd block = cz.block(rs, cs, cur_cell_size, cur_cell_size);
                    MatrixXd::Index ri, ci;
                    block.minCoeff(&ri, &ci);

                    ri += rs;
                    ci += cs;

                    dcz(r, c) = surface(rr, cc);
                    cz(ri, ci) = surface(rr, cc);
                }
            }
        }

        if (log()->getLevel() > LogLevel::Debug5)
        {
            char buffer[256];
            sprintf(buffer, "surface_%d.tif", l);
            std::string name(buffer);
            writeMatrix(surface, name, cur_cell_size, view);

            char bufm[256];
            sprintf(bufm, "master_control_%d.laz", l);
            std::string namem(bufm);
            writeControl(cx, cy, cz, namem);

            // this is identical to filtered control when written here - should move it...
            char buf3[256];
            sprintf(buf3, "control_%d.laz", l);
            std::string name3(buf3);
            writeControl(dcx, dcy, dcz, name3);

            char rbuf[256];
            sprintf(rbuf, "residual_%d.tif", l);
            std::string rbufn(rbuf);
            writeMatrix(R, rbufn, cur_cell_size, view);

            char obuf[256];
            sprintf(obuf, "open_%d.tif", l);
            std::string obufn(obuf);
            writeMatrix(maxZ, obufn, cur_cell_size, view);

            char Tbuf[256];
            sprintf(Tbuf, "tophat_%d.tif", l);
            std::string Tbufn(Tbuf);
            writeMatrix(T, Tbufn, cur_cell_size, view);

            char tbuf[256];
            sprintf(tbuf, "thresh_%d.tif", l);
            std::string tbufn(tbuf);
            writeMatrix(t, tbufn, cur_cell_size, view);

            char buf2[256];
            sprintf(buf2, "filtered_control_%d.laz", l);
            std::string name2(buf2);
            writeControl(dcx, dcy, dcz, name2);
        }
    }

    MatrixXd surface = TPS(dcx, dcy, dcz, cur_cell_size);
    MatrixXd R = computeResidual(cz, surface);
    MatrixXd maxZ = matrixOpen(R, 2);
    MatrixXd T = R - maxZ;
    MatrixXd t = computeThresholds(T, 2);

    if (log()->getLevel() > LogLevel::Debug5)
    {
        writeControl(cx, cy, mc, "closed.laz");

        char buffer[256];
        sprintf(buffer, "final_surface.tif");
        std::string name(buffer);
        writeMatrix(surface, name, cur_cell_size, view);

        char rbuf[256];
        sprintf(rbuf, "final_residual.tif");
        std::string rbufn(rbuf);
        writeMatrix(R, rbufn, cur_cell_size, view);

        char obuf[256];
        sprintf(obuf, "final_opened.tif");
        std::string obufn(obuf);
        writeMatrix(maxZ, obufn, cur_cell_size, view);

        char Tbuf[256];
        sprintf(Tbuf, "final_tophat.tif");
        std::string Tbufn(Tbuf);
        writeMatrix(T, Tbufn, cur_cell_size, view);

        char tbuf[256];
        sprintf(tbuf, "final_thresh.tif");
        std::string tbufn(tbuf);
        writeMatrix(t, tbufn, cur_cell_size, view);
    }

    // apply final filtering (top hat) using raw points against TPS

    // ...the LiDAR points are filtered only at the bottom level.
    for (auto i = 0; i < np; ++i)
    {
        using namespace Dimension;
        
        double x = view->getFieldAs<double>(Id::X, i);
        double y = view->getFieldAs<double>(Id::Y, i);
        double z = view->getFieldAs<double>(Id::Z, i);

        int c = clamp(getColIndex(x, cur_cell_size), 0, m_numCols-1);
        int r = clamp(getRowIndex(y, cur_cell_size), 0, m_numRows-1);

        double res = z - surface(r, c);
        if (res < t(r, c))
            groundIdx.push_back(i);
    }
    std::cerr << "done\n";
    std::cerr << groundIdx.size() << std::endl;

    return groundIdx;
}

void MongusFilter::downsampleMin(Eigen::MatrixXd *cx, Eigen::MatrixXd *cy,
                                 Eigen::MatrixXd* cz, Eigen::MatrixXd *dcx,
                                 Eigen::MatrixXd *dcy, Eigen::MatrixXd* dcz,
                                 double cell_size)
{
    int nr = ceil(cz->rows() / cell_size);
    int nc = ceil(cz->cols() / cell_size);

    std::cerr << nr << "\t" << nc << "\t" << cell_size << std::endl;

    dcx->resize(nr, nc);
    dcx->setConstant(std::numeric_limits<double>::quiet_NaN());

    dcy->resize(nr, nc);
    dcy->setConstant(std::numeric_limits<double>::quiet_NaN());

    dcz->resize(nr, nc);
    dcz->setConstant(std::numeric_limits<double>::max());

    for (auto c = 0; c < cz->cols(); ++c)
    {
        for (auto r = 0; r < cz->rows(); ++r)
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

Eigen::MatrixXd MongusFilter::computeResidual(Eigen::MatrixXd cz,
        Eigen::MatrixXd surface)
{
    using namespace Eigen;

    MatrixXd R = MatrixXd::Zero(cz.rows(), cz.cols());
    for (auto c = 0; c < cz.cols(); ++c)
    {
        for (auto r = 0; r < cz.rows(); ++r)
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

Eigen::MatrixXd MongusFilter::padMatrix(Eigen::MatrixXd d, int r)
{
    using namespace Eigen;

    MatrixXd out = MatrixXd::Zero(d.rows()+2*r, d.cols()+2*r);
    out.block(r, r, d.rows(), d.cols()) = d;
    out.block(r, 0, d.rows(), r) =
        d.block(0, 0, d.rows(), r).rowwise().reverse();
    out.block(r, d.cols()+r, d.rows(), r) =
        d.block(0, d.cols()-r, d.rows(), r).rowwise().reverse();
    out.block(0, 0, r, out.cols()) =
        out.block(r, 0, r, out.cols()).colwise().reverse();
    out.block(d.rows()+r, 0, r, out.cols()) =
        out.block(out.rows()-r, 0, r, out.cols()).colwise().reverse();

    return out;
}

Eigen::MatrixXd MongusFilter::matrixOpen(Eigen::MatrixXd data, int radius)
{
    using namespace Eigen;

    MatrixXd data2 = padMatrix(data, radius);

    int nrows = data2.rows();
    int ncols = data2.cols();

    // first min, then max of min
    MatrixXd minZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::max());
    MatrixXd maxZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::lowest());
    for (auto c = 0; c < ncols; ++c)
    {
        for (auto r = 0; r < nrows; ++r)
        {
            int cs = clamp(c-radius, 0, ncols-1);
            int ce = clamp(c+radius, 0, ncols-1);
            int rs = clamp(r-radius, 0, nrows-1);
            int re = clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (data2(row, col) < minZ(r, c))
                        minZ(r, c) = data2(row, col);
                }
            }
        }
    }
    for (auto c = 0; c < ncols; ++c)
    {
        for (auto r = 0; r < nrows; ++r)
        {
            int cs = clamp(c-radius, 0, ncols-1);
            int ce = clamp(c+radius, 0, ncols-1);
            int rs = clamp(r-radius, 0, nrows-1);
            int re = clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (minZ(row, col) > maxZ(r, c))
                        maxZ(r, c) = minZ(row, col);
                }
            }
        }
    }

    return maxZ.block(radius, radius, data.rows(), data.cols());
}

Eigen::MatrixXd MongusFilter::matrixClose(Eigen::MatrixXd data, int radius)
{
    using namespace Eigen;

    MatrixXd data2 = padMatrix(data, radius);

    int nrows = data2.rows();
    int ncols = data2.cols();

    // first min, then max of min
    MatrixXd minZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::max());
    MatrixXd maxZ = MatrixXd::Constant(nrows, ncols,
                                       std::numeric_limits<double>::lowest());
    for (auto c = 0; c < ncols; ++c)
    {
        for (auto r = 0; r < nrows; ++r)
        {
            int cs = clamp(c-radius, 0, ncols-1);
            int ce = clamp(c+radius, 0, ncols-1);
            int rs = clamp(r-radius, 0, nrows-1);
            int re = clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (data2(row, col) > maxZ(r, c))
                        maxZ(r, c) = data2(row, col);
                }
            }
        }
    }
    for (auto c = 0; c < ncols; ++c)
    {
        for (auto r = 0; r < nrows; ++r)
        {
            int cs = clamp(c-radius, 0, ncols-1);
            int ce = clamp(c+radius, 0, ncols-1);
            int rs = clamp(r-radius, 0, nrows-1);
            int re = clamp(r+radius, 0, nrows-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    if (maxZ(row, col) < minZ(r, c))
                        minZ(r, c) = maxZ(row, col);
                }
            }
        }
    }

    return minZ.block(radius, radius, data.rows(), data.cols());
}

Eigen::MatrixXd MongusFilter::computeThresholds(Eigen::MatrixXd T, int radius)
{
    using namespace Eigen;

    MatrixXd t = MatrixXd::Zero(T.rows(), T.cols());
    for (auto c = 0; c < T.cols(); ++c)
    {
        for (auto r = 0; r < T.rows(); ++r)
        {
            double M1 = 0;
            double M2 = 0;
            int n = 0;

            int cs = clamp(c-radius, 0, T.cols()-1);
            int ce = clamp(c+radius, 0, T.cols()-1);
            int rs = clamp(r-radius, 0, T.rows()-1);
            int re = clamp(r+radius, 0, T.rows()-1);

            for (auto col = cs; col <= ce; ++col)
            {
                for (auto row = rs; row <= re; ++row)
                {
                    if ((row-r)*(row-r)+(col-c)*(col-c) > radius*radius)
                        continue;
                    int n1 = n;
                    n++;
                    double delta = T(row, col) - M1;
                    double delta_n = delta / n;
                    double term1 = delta * delta_n * n1;
                    M1 += delta_n;
                    M2 += term1;
                }
            }
            t(r, c) = M1 + m_k * std::sqrt(M2/(n-1));
        }
    }

    return t;
}

PointViewSet MongusFilter::run(PointViewPtr view)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process MongusFilter...\n";

    std::vector<PointId> idx = processGround(view);
    std::cerr << idx.size() << std::endl;

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
