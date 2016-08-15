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

#include <pdal/KDIndex.hpp>
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

void MongusFilter::computeSpline(PointViewPtr view, std::vector<PointId> control, Eigen::Ref<Eigen::Vector3d> a, Eigen::Ref<Eigen::VectorXd> w, std::map<PointId, std::vector<PointId> > mymap)
{
    using namespace Dimension;
    using namespace Eigen;
    
    log()->get(LogLevel::Debug) << "Computing spline for " << control.size() << " samples\n";
    
    auto sqrDist = [](double xi, double xj, double yi, double yj)
    {
        return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
    };

    VectorXd T = VectorXd::Zero(control.size());
    MatrixXd P = MatrixXd::Zero(control.size(), 3);
    MatrixXd K = MatrixXd::Zero(control.size(), control.size());
    
    for (auto i = 0; i < control.size(); ++i)
    {
        PointId ii = control[i];
        double xi = view->getFieldAs<double>(Id::X, ii);
        double yi = view->getFieldAs<double>(Id::Y, ii);
        double zi = view->getFieldAs<double>(Id::Z, ii);
        T(i) = zi;
        P.row(i) << 1, xi, yi;
        
        std::vector<PointId> neighbors = mymap[ii];
        for (auto j = 0; j < neighbors.size(); ++j)
        {
            PointId jj = neighbors[j];
            if (ii == jj)
                continue;
            double xj = view->getFieldAs<double>(Id::X, jj);
            double yj = view->getFieldAs<double>(Id::Y, jj);
            double rsqr = sqrDist(xi, xj, yi, yj);
            K(i, j) = rsqr * std::log10(std::sqrt(rsqr));
        }
    }
    
    MatrixXd A = MatrixXd::Zero(control.size()+3, control.size()+3);
    A.block(0,0,control.size(),control.size()) = K;
    A.block(0,control.size(),control.size(),3) = P;
    A.block(control.size(),0,3,control.size()) = P.transpose();

    VectorXd b = VectorXd::Zero(control.size()+3);
    b.head(control.size()) = T;

    VectorXd x = A.colPivHouseholderQr().solve(b);

    a = x.tail(3);
    w = x.head(control.size());
}

std::vector<PointId> MongusFilter::interpolateSpline(PointViewPtr view,
        std::vector<PointId> samples,
        std::vector<PointId> control, std::map<PointId, std::vector<PointId> > mymap)
{
    using namespace Dimension;
    using namespace Eigen;

    // for each sx, sy sample location, compute the interpolate the value Z from spline control points cx, cy, cz
    
    std::vector<PointId> newcontrol;

    std::vector<double> z(samples.size());
    std::vector<double> residuals(samples.size());
    
    auto sqrDist = [](double xi, double xj, double yi, double yj)
    {
        return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
    };

    for (auto k = 0; k < samples.size(); ++k)
    {
        PointId kk = samples[k];

        double xk = view->getFieldAs<double>(Id::X, kk);
        double yk = view->getFieldAs<double>(Id::Y, kk);
        
        // only for debugging
        double zk = view->getFieldAs<double>(Id::Z, kk);
        // log()->get(LogLevel::Debug) << "Interpolating spline at (" << xk << "," << yk << "," << zk << ")...\n";
        
        std::vector<PointId> neighbors = mymap[kk];
        
        // log()->get(LogLevel::Debug) << "interpolateSpline: Computing spline with " << neighbors.size() << " neigbors\n";

        VectorXd T = VectorXd::Zero(neighbors.size());
        MatrixXd P = MatrixXd::Zero(neighbors.size(), 3);
        MatrixXd K = MatrixXd::Zero(neighbors.size(), neighbors.size());
        
        for (auto i = 0; i < neighbors.size(); ++i)
        {
            PointId ii = neighbors[i];
            double xi = view->getFieldAs<double>(Id::X, ii);
            double yi = view->getFieldAs<double>(Id::Y, ii);
            double zi = view->getFieldAs<double>(Id::Z, ii);
            T(i) = zi;
            P.row(i) << 1, xi, yi;
            
            for (auto j = 0; j < neighbors.size(); ++j)
            {
                PointId jj = neighbors[j];
                if (ii == jj)
                    continue;
                double xj = view->getFieldAs<double>(Id::X, jj);
                double yj = view->getFieldAs<double>(Id::Y, jj);
                double rsqr = sqrDist(xi, xj, yi, yj);
                K(i, j) = rsqr * std::log10(std::sqrt(rsqr));
            }
        }
        
        MatrixXd A = MatrixXd::Zero(neighbors.size()+3, neighbors.size()+3);
        A.block(0,0,neighbors.size(),neighbors.size()) = K;
        A.block(0,neighbors.size(),neighbors.size(),3) = P;
        A.block(neighbors.size(),0,3,neighbors.size()) = P.transpose();

        VectorXd b = VectorXd::Zero(neighbors.size()+3);
        b.head(neighbors.size()) = T;

        VectorXd x = A.colPivHouseholderQr().solve(b);

        Vector3d a = x.tail(3);
        VectorXd w = x.head(neighbors.size());

        double sum = 0.0;
        for (auto j = 0; j < neighbors.size(); ++j)
        {
            PointId jj = neighbors[j];
            if (kk == jj)
                continue;
            double xj = view->getFieldAs<double>(Id::X, jj);
            double yj = view->getFieldAs<double>(Id::Y, jj);
            double rsqr = sqrDist(xk, xj, yk, yj);
            // log()->get(LogLevel::Debug) << "rsqr = " << rsqr << std::endl;
            // log()->get(LogLevel::Debug) << "log(sqrt(rsqr)) = " << std::log10(std::sqrt(rsqr)) << std::endl;
            // log()->get(LogLevel::Debug) << "w = " << w(j) << std::endl;
            
            sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
            // log()->get(LogLevel::Debug) << "sum = " << sum << std::endl;
        }

        z[k] = a(0) + a(1)*xk + a(2)*yk + sum;
        // log()->get(LogLevel::Debug) << "\t..." << z[k] << "\t" << zk - z[k] << std::endl;
        residuals[k] = zk - z[k];
    }
    
    int n = 1;
    double M1 = 0.0;
    double M2 = 0.0;
    for (auto const& r : residuals)
    {
        double delta, delta_n, delta_n2, term1;
        int n1 = n;
        delta = r - M1;
        delta_n = delta / n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        M1 += delta_n;
        M2 += term1;
        n++;
    }
    double mean = M1;
    double std = std::sqrt(M2 / (residuals.size()-1.0));
    double t1 = M1 - 1*std;
    double t2 = M1 + 1*std;
    log()->get(LogLevel::Debug) << mean << "\t" << std << "\t" << t1 << "\t" << t2 << std::endl;
    
    for (auto k = 0; k < samples.size(); ++k)
    {
        if (residuals[k] < t1 || residuals[k] > t2)
            continue;
        newcontrol.push_back(samples[k]);
    }
    return newcontrol;
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

void MongusFilter::writeSurface(std::string filename, PointViewPtr view, std::vector<PointId> control, std::map<PointId, std::vector<PointId> > mymap)
{
  // std::cerr << w.transpose() << std::endl;
  // std::cerr << w.size() << std::endl;
    int cols = m_numCols;
    int rows = m_numRows;

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
            adfGeoTransform[1] = m_cellSize;
            adfGeoTransform[2] = 0.0;
            adfGeoTransform[3] = m_bounds.maxy; // + 0.5*m_GRID_DIST_Y;
            adfGeoTransform[4] = 0.0;
            adfGeoTransform[5] = -1 * m_cellSize;
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
        
        using namespace Dimension;
        using namespace Eigen;

        // for each sx, sy sample location, compute the interpolate the value Z from spline control points cx, cy, cz
        
        auto sqrDist = [](double xi, double xj, double yi, double yj)
        {
            return (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
        };
        
        PointViewPtr ctrl = view->makeNew();
        for (auto const& i : control)
        {
            ctrl->appendPoint(*view, i);
        }
        
        // Build the 2D KD-tree.
        KD2Index kd(*ctrl);
        kd.build();
        
        // VectorXd T = VectorXd::Zero(control.size());
        // MatrixXd P = MatrixXd::Zero(control.size(), 3);
        // MatrixXd K = MatrixXd::Zero(control.size(), control.size());
        // 
        // for (auto i = 0; i < control.size(); ++i)
        // {
        //     PointId ii = control[i];
        //     double xi = view->getFieldAs<double>(Id::X, ii);
        //     double yi = view->getFieldAs<double>(Id::Y, ii);
        //     double zi = view->getFieldAs<double>(Id::Z, ii);
        //     T(i) = zi;
        //     P.row(i) << 1, xi, yi;
        //     for (auto j = 0; j < control.size(); ++j)
        //     {
        //         PointId jj = control[j];
        //         if (ii == jj)
        //             continue;
        //         double xj = view->getFieldAs<double>(Id::X, jj);
        //         double yj = view->getFieldAs<double>(Id::Y, jj);
        //         double rsqr = sqrDist(xi, xj, yi, yj);
        //         K(i, j) = rsqr * std::log10(std::sqrt(rsqr));
        //     }
        // }
        // // log()->get(LogLevel::Debug) << "T: " << T.transpose() << std::endl;
        // // log()->get(LogLevel::Debug) << "P: " << P.transpose() << std::endl;
        // // log()->get(LogLevel::Debug) << "K: " << K << std::endl;
        // 
        // MatrixXd A = MatrixXd::Zero(control.size()+3, control.size()+3);
        // A.block(0,0,control.size(),control.size()) = K;
        // A.block(0,control.size(),control.size(),3) = P;
        // A.block(control.size(),0,3,control.size()) = P.transpose();
        // 
        // VectorXd b = VectorXd::Zero(control.size()+3);
        // b.head(control.size()) = T;
        // 
        // VectorXd x = A.fullPivHouseholderQr().solve(b);
        // 
        // Vector3d a = x.tail(3);
        // VectorXd w = x.head(control.size());
        
        // ArrayXd xarr(control.size());
        // ArrayXd yarr(control.size());
        // for (auto j = 0; j < control.size(); ++j)
        // {
        //     PointId jj = control[j];
        //     xarr(j) = view->getFieldAs<double>(Id::X, jj);
        //     yarr(j) = view->getFieldAs<double>(Id::Y, jj);
        // }
        // 
        // ArrayXd warr = w.array();
        
        #pragma omp parallel for
        for (auto c = cs; c < ce; ++c)
        {
            double xk = m_bounds.minx + (c+0.5)*m_cellSize;
            // ArrayXd xdiff = xk - xarr;
            // log()->get(LogLevel::Debug) << "xd: " << xdiff << std::endl;
            // ArrayXd xdiff2 = xdiff * xdiff;
                  
            for (auto r = rs; r < re; ++r)
            {
              // std::cerr << warr.size() << std::endl;
              // std::cerr << w.array().size() << std::endl;
              // log()->get(LogLevel::Debug) << "w: " << warr << std::endl;
              
                  double yk = m_maxRow - (r+0.5)*m_cellSize;
                  // ArrayXd ydiff = yk - yarr;
                  
                  std::vector<PointId> nearest = kd.neighbors(xk, yk, 1);
                  
                  std::vector<PointId> neighbors = mymap[control[nearest[0]]];

                  VectorXd T = VectorXd::Zero(neighbors.size());
                  MatrixXd P = MatrixXd::Zero(neighbors.size(), 3);
                  MatrixXd K = MatrixXd::Zero(neighbors.size(), neighbors.size());
                  
                  for (auto i = 0; i < neighbors.size(); ++i)
                  {
                      PointId ii = neighbors[i];
                      double xi = view->getFieldAs<double>(Id::X, ii);
                      double yi = view->getFieldAs<double>(Id::Y, ii);
                      double zi = view->getFieldAs<double>(Id::Z, ii);
                      T(i) = zi;
                      P.row(i) << 1, xi, yi;
                      
                      for (auto j = 0; j < neighbors.size(); ++j)
                      {
                          PointId jj = neighbors[j];
                          if (ii == jj)
                              continue;
                          double xj = view->getFieldAs<double>(Id::X, jj);
                          double yj = view->getFieldAs<double>(Id::Y, jj);
                          double rsqr = sqrDist(xi, xj, yi, yj);
                          K(i, j) = rsqr * std::log10(std::sqrt(rsqr));
                      }
                  }
                  
                  MatrixXd A = MatrixXd::Zero(neighbors.size()+3, neighbors.size()+3);
                  A.block(0,0,neighbors.size(),neighbors.size()) = K;
                  A.block(0,neighbors.size(),neighbors.size(),3) = P;
                  A.block(neighbors.size(),0,3,neighbors.size()) = P.transpose();

                  VectorXd b = VectorXd::Zero(neighbors.size()+3);
                  b.head(neighbors.size()) = T;

                  VectorXd x = A.colPivHouseholderQr().solve(b);

                  Vector3d a = x.tail(3);
                  VectorXd w = x.head(neighbors.size());
                  
                  // log()->get(LogLevel::Debug) << "yk: " << yk << std::endl;
                  // log()->get(LogLevel::Debug) << "yarr: " << yarr << std::endl;
                  // log()->get(LogLevel::Debug) << "yd: " << ydiff << std::endl;
                  // ArrayXd ydiff2 = ydiff * ydiff;
                  // ArrayXd rsqr = xdiff2 + ydiff2;
                  // ArrayXd parts = w.array() * rsqr * rsqr.sqrt().log();
                  // double sum = parts.sum();
                  
                  // log()->get(LogLevel::Debug) << "rsqr: " << rsqr << std::endl;
                  // log()->get(LogLevel::Debug) << "logrsqr: " << rsqr.sqrt().log() << std::endl;
                  // log()->get(LogLevel::Debug) << "rsqrtlogrsqr: " << rsqr*rsqr.sqrt().log() << std::endl;
                  // log()->get(LogLevel::Debug) << "parts: " << parts << std::endl;
                  // log()->get(LogLevel::Debug) << "sum: " <<  sum << std::endl;
                  
                  
                  double sum = 0.0;
                  for (auto j = 0; j < neighbors.size(); ++j)
                  {
                      PointId jj = neighbors[j];
                      double xj = view->getFieldAs<double>(Id::X, jj);
                      double yj = view->getFieldAs<double>(Id::Y, jj);
                      double rsqr = sqrDist(xk, xj, yk, yj);
                      sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
                  }
              
                poRasterData[(r * cols) + c] = a(0) + a(1)*xk + a(2)*yk + sum;
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
    // MatrixXd cx(m_numRows, m_numCols);
    // cx.setConstant(std::numeric_limits<double>::quiet_NaN());
    // 
    // MatrixXd cy(m_numRows, m_numCols);
    // cy.setConstant(std::numeric_limits<double>::quiet_NaN());
    
    MatrixXi ci(m_numRows, m_numCols);
    ci.setConstant(std::numeric_limits<int>::quiet_NaN());

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
            cz(r, c) = z;
            ci(r, c) = i;
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
        {
            cz(i) = std::numeric_limits<double>::max();
            ci(i) = std::numeric_limits<int>::quiet_NaN();
        }
            // cz(i) = mc(i);
    }
    
    auto h = buildLevels(ci, cz);
    log()->get(LogLevel::Debug) << "hierarchy supposedly build\n";

    // downsample control at max_level
    int level = m_l;
    double cur_cell_size = m_cellSize * std::pow(2, level-1);

    // MatrixXd dcx, dcy, dcz;

    // Top-level control samples are assumed to be ground points, no filtering
    // is applied.
    // std::vector<PointId> control = downsampleMin(ci, cz, level);
    std::map<PointId, std::vector<PointId> > h0 = h[m_l];
    std::cerr << h0.size() << " control at level 0\n";
    std::vector<PointId> control(h0.size());
    int i = 0;
    for (auto const& hh : h0)
    {
        control[i++] = hh.first;
    }
    log()->get(LogLevel::Debug) << control.size() << " control indices at level " << level << std::endl;

    // Point-filtering is performed iteratively at each level of the
    // control-points hierarchy in a top-down fashion
    for (auto l = level-1; l > 0; --l)
    {
        std::cerr << "Level " << l << std::endl;

        // compute TPS with update control at level

        // downsample control at level
        cur_cell_size /= 2;
        // std::vector<PointId> samples = downsampleMin(ci, cz, l);
        std::map<PointId, std::vector<PointId> > hl = h[l];
        std::vector<PointId> samples(hl.size());
        int i = 0;
        for (auto const& hh : hl)
        {
            samples[i++] = hh.first;
        }
        log()->get(LogLevel::Debug) << samples.size() << " samples indices at level " << l << std::endl;
        
        // The interpolated surface is estimated based on the filtered set of
        // TPS control-points at the previous level of hierarchy
        // MatrixXd surface = TPS(dcx, dcy, dcz, cur_cell_size);
        // Vector3d a = Vector3d::Zero();
        // VectorXd w = VectorXd::Zero(control.size());
        // computeSpline(view, control, a, w, h[l]);
        std::vector<PointId> newcontrol = interpolateSpline(view, samples, control, h[l]);
        // std::cerr << a.transpose() << std::endl;
        // std::cerr << w.transpose() << std::endl;
        log()->get(LogLevel::Debug) << newcontrol.size() << " samples are kept for the next iteration\n";
        
        control.swap(newcontrol);
        
        char buf[256];
        sprintf(buf, "surf_%d.tif", l);
        std::string name(buf);
        // Vector3d newa = Vector3d::Zero();
        // VectorXd neww = VectorXd::Zero(control.size());
        // computeSpline(view, control, newa, neww, h[l]);
        // writeSurface(name, view, control, h[l]);
        
        // MatrixXd R = computeResidual(dcz, surface);
        // MatrixXd maxZ = matrixOpen(R, 2*l);
        // MatrixXd T = R - maxZ;
        // MatrixXd t = computeThresholds(T, 2*l);
        // 
        // // the time complexity of the approach is reduced by filtering only the
        // // control-points in each iteration
        // for (auto c = 0; c < T.cols(); ++c)
        // {
        //     for (auto r = 0; r < T.rows(); ++r)
        //     {
        //         // If the TPS control-point is recognized as a non-ground point,
        //         // it is replaced by the interpolated point.
        //         if (T(r,c) > t(r,c))
        //         {
        //             int rr = std::floor(r/2);
        //             int cc = std::floor(c/2);
        //             int rs = cur_cell_size * r;
        //             int cs = cur_cell_size * c;
        //             MatrixXd block = cz.block(rs, cs, cur_cell_size, cur_cell_size);
        //             MatrixXd::Index ri, ci;
        //             block.minCoeff(&ri, &ci);
        // 
        //             ri += rs;
        //             ci += cs;
        // 
        //             dcz(r, c) = surface(rr, cc);
        //             cz(ri, ci) = surface(rr, cc);
        //         }
        //     }
        // }
        // 
        // if (log()->getLevel() > LogLevel::Debug5)
        // {
        //     char buffer[256];
        //     sprintf(buffer, "surface_%d.tif", l);
        //     std::string name(buffer);
        //     writeMatrix(surface, name, cur_cell_size, view);
        // 
        //     char bufm[256];
        //     sprintf(bufm, "master_control_%d.laz", l);
        //     std::string namem(bufm);
        //     // writeControl(cx, cy, cz, namem);
        // 
        //     // this is identical to filtered control when written here - should move it...
        //     char buf3[256];
        //     sprintf(buf3, "control_%d.laz", l);
        //     std::string name3(buf3);
        //     // writeControl(dcx, dcy, dcz, name3);
        // 
        //     char rbuf[256];
        //     sprintf(rbuf, "residual_%d.tif", l);
        //     std::string rbufn(rbuf);
        //     writeMatrix(R, rbufn, cur_cell_size, view);
        // 
        //     char obuf[256];
        //     sprintf(obuf, "open_%d.tif", l);
        //     std::string obufn(obuf);
        //     writeMatrix(maxZ, obufn, cur_cell_size, view);
        // 
        //     char Tbuf[256];
        //     sprintf(Tbuf, "tophat_%d.tif", l);
        //     std::string Tbufn(Tbuf);
        //     writeMatrix(T, Tbufn, cur_cell_size, view);
        // 
        //     char tbuf[256];
        //     sprintf(tbuf, "thresh_%d.tif", l);
        //     std::string tbufn(tbuf);
        //     writeMatrix(t, tbufn, cur_cell_size, view);
        // 
        //     char buf2[256];
        //     sprintf(buf2, "filtered_control_%d.laz", l);
        //     std::string name2(buf2);
        //     // writeControl(dcx, dcy, dcz, name2);
        // }
    }

    // MatrixXd surface = TPS(dcx, dcy, dcz, cur_cell_size);
    // MatrixXd R = computeResidual(cz, surface);
    // MatrixXd maxZ = matrixOpen(R, 2);
    // MatrixXd T = R - maxZ;
    // MatrixXd t = computeThresholds(T, 2);
    // 
    // if (log()->getLevel() > LogLevel::Debug5)
    // {
    //     // writeControl(cx, cy, mc, "closed.laz");
    // 
    //     char buffer[256];
    //     sprintf(buffer, "final_surface.tif");
    //     std::string name(buffer);
    //     writeMatrix(surface, name, cur_cell_size, view);
    // 
    //     char rbuf[256];
    //     sprintf(rbuf, "final_residual.tif");
    //     std::string rbufn(rbuf);
    //     writeMatrix(R, rbufn, cur_cell_size, view);
    // 
    //     char obuf[256];
    //     sprintf(obuf, "final_opened.tif");
    //     std::string obufn(obuf);
    //     writeMatrix(maxZ, obufn, cur_cell_size, view);
    // 
    //     char Tbuf[256];
    //     sprintf(Tbuf, "final_tophat.tif");
    //     std::string Tbufn(Tbuf);
    //     writeMatrix(T, Tbufn, cur_cell_size, view);
    // 
    //     char tbuf[256];
    //     sprintf(tbuf, "final_thresh.tif");
    //     std::string tbufn(tbuf);
    //     writeMatrix(t, tbufn, cur_cell_size, view);
    // }

    // apply final filtering (top hat) using raw points against TPS

    // // ...the LiDAR points are filtered only at the bottom level.
    // for (auto i = 0; i < np; ++i)
    // {
    //     using namespace Dimension;
    // 
    //     double x = view->getFieldAs<double>(Id::X, i);
    //     double y = view->getFieldAs<double>(Id::Y, i);
    //     double z = view->getFieldAs<double>(Id::Z, i);
    // 
    //     int c = clamp(getColIndex(x, cur_cell_size), 0, m_numCols-1);
    //     int r = clamp(getRowIndex(y, cur_cell_size), 0, m_numRows-1);
    // 
    //     double res = z - surface(r, c);
    //     if (res < t(r, c))
    //         groundIdx.push_back(i);
    // }
    
    groundIdx.swap(control);

    return groundIdx;
}

std::map<int, std::map<PointId, std::vector<PointId> > > MongusFilter::buildLevels(Eigen::MatrixXi ci, Eigen::MatrixXd cz)
{
    using namespace Eigen;
    
    std::map<int, std::map<PointId, std::vector<PointId> > > hierarchy;
    for (auto level = 0; level <= m_l; ++level)
    {
        MatrixXi localCi = ci;
        if (level > 0)
        {
            int step = std::pow(2, level);
            // localCi.resize(std::ceil(cz.rows()/step), std::ceil(cz.cols()/step));
            log()->get(LogLevel::Debug) << "Downsampling at level " << level
                                        << " with step size of " << step << std::endl;
            log()->get(LogLevel::Debug) << localCi.rows() << "\t" << localCi.cols() << std::endl;
                                        
            for (auto c = 0; c < cz.cols(); c+=step)
            {
                for (auto r = 0; r < cz.rows(); r+=step)
                {
                    int re = clamp(r+step, 0, cz.rows());
                    int ce = clamp(c+step, 0, cz.cols());
                    int rsize = re - r;
                    int csize = ce - c;
                    
                    MatrixXd::Index minRow, minCol;
                    cz.block(r, c, rsize, csize).minCoeff(&minRow, &minCol);
                    int rr = std::floor(r / step);
                    int cc = std::floor(c / step);
                    localCi(rr, cc) = ci.block(r, c, rsize, csize)(minRow, minCol);
                }
            }
            localCi.conservativeResize(std::ceil(cz.rows()/step),std::ceil(cz.cols()/step));
        }
        
        std::map<PointId, std::vector<PointId> > mymap;
        for (auto c = 0; c < localCi.cols(); ++c)
        {
            for (auto r = 0; r < localCi.rows(); ++r)
            {
                // insert into map the pair of PointId at ci(r, c) to neighbors in +/- 3 cells at current level
                int rs = clamp(r-3, 0, localCi.rows());
                int re = clamp(r+3, 0, localCi.rows());
                int rsize = re-rs;
                int cs = clamp(c-3, 0, localCi.cols());
                int ce = clamp(c+3, 0, localCi.cols());
                int csize = ce-cs;
                MatrixXi b = localCi.block(rs, cs, rsize, csize); // these are the neighbors or ci(r, c)
                std::vector<PointId> neighbors(b.size());
                for (auto i = 0; i < b.size(); ++i)
                {
                    neighbors[i] = b(i);
                }
                PointId key = localCi(r, c);
                mymap[key] = neighbors;
            }
        }
        
        hierarchy[level] = mymap;
    }
    
    return hierarchy;
}

std::vector<PointId> MongusFilter::downsampleMin(Eigen::MatrixXi ci,
                                 Eigen::MatrixXd cz,
                                 int level)
{
    using namespace Eigen;
    
    assert(ci.rows() == cz.rows());
    assert(ci.cols() == cz.cols());
    
    int step = std::pow(2, level);
    log()->get(LogLevel::Debug) << "Downsampling at level " << level
                                << " with step size of " << step << std::endl;
    std::vector<PointId> ids;

    for (auto c = 0; c < cz.cols(); c+=step)
    {
        for (auto r = 0; r < cz.rows(); r+=step)
        {
            int re = clamp(r+step, 0, cz.rows());
            int ce = clamp(c+step, 0, cz.cols());
            int rsize = re - r;
            int csize = ce - c;
            // log()->get(LogLevel::Debug) << "row: " << r << "\t" << re << "\t" << rsize << std::endl;
            // log()->get(LogLevel::Debug) << "col: " << c << "\t" << ce << "\t" << csize << std::endl;
            MatrixXd::Index minRow, minCol;
            cz.block(r, c, rsize, csize).minCoeff(&minRow, &minCol);
            ids.push_back(ci.block(r, c, rsize, csize)(minRow, minCol));
            // log()->get(LogLevel::Debug) << cz.block(r, c, rsize, csize)(minRow, minCol) << "\t" << ci.block(r, c, rsize, csize)(minRow, minCol) << std::endl;
        }
    }
    
    return ids;
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
