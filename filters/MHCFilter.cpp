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

#include "MHCFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>
#include <pdal/EigenUtils.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/SpatialReference.hpp>
#include <io/BufferReader.hpp>
#include <pdal/util/FileUtils.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#include <Eigen/Dense>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.mhc",
        "Multiresolution Hierarchical Classification (Chen et al., 2013)",
        "http://pdal.io/stages/filters.mhc.html");

CREATE_STATIC_PLUGIN(1, 0, MHCFilter, Filter, s_info)

std::string MHCFilter::getName() const
{
    return s_info.name;
}

void MHCFilter::addArgs(ProgramArgs& args)
{
    args.add("h", "Initial resolution", m_res, 2.0);
    args.add("i", "Initial threshold", m_thresh, 0.5);
    args.add("classify", "Apply classification labels?", m_classify, true);
    args.add("extract", "Extract ground returns?", m_extract);
    args.add("outdir", "Optional output directory for debugging", m_outDir);
}

void MHCFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
}

Eigen::MatrixXd MHCFilter::computeSpline(PointViewPtr view, size_t rows,
    size_t cols, double res, BOX2D bounds)
{
    using namespace Eigen;
    
    KD2Index index(*view);
    index.build();

    MatrixXd S = MatrixXd::Zero(rows, cols);

    for (size_t col = 0; col < cols; ++col)
    {
        for (size_t row = 0; row < rows; ++row)
        {
            double xi2 = bounds.minx + (col + 0.5) * res;
            double yi2 = bounds.miny + (row + 0.5) * res;
            
            int nsize = 12;
            std::vector<PointId> idx = index.neighbors(xi2, yi2, nsize);
          
            VectorXd T = VectorXd::Zero(nsize);
            MatrixXd P = MatrixXd::Zero(nsize, 3);
            MatrixXd K = MatrixXd::Zero(nsize, nsize);

            for (auto id = 0; id < nsize; ++id)
            {
                double xj = view->getFieldAs<double>(Dimension::Id::X, idx[id]);
                double yj = view->getFieldAs<double>(Dimension::Id::Y, idx[id]);
                double zj = view->getFieldAs<double>(Dimension::Id::Z, idx[id]);
                T(id) = zj;
                P.row(id) << 1, xj, yj;
                for (auto id2 = 0; id2 < nsize; ++id2)
                {
                    if (id == id2)
                        continue;
                    double xk = view->getFieldAs<double>(Dimension::Id::X, idx[id2]);
                    double yk = view->getFieldAs<double>(Dimension::Id::Y, idx[id2]);
                    double zk = view->getFieldAs<double>(Dimension::Id::Z, idx[id2]);
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
            for (auto j = 0; j < nsize; ++j)
            {
                double xj = view->getFieldAs<double>(Dimension::Id::X, idx[j]);
                double yj = view->getFieldAs<double>(Dimension::Id::Y, idx[j]);
                double zj = view->getFieldAs<double>(Dimension::Id::Z, idx[j]);
                double rsqr = (xj - xi2) * (xj - xi2) + (yj - yi2) * (yj - yi2);
                if (rsqr == 0.0)
                    continue;
                sum += w(j) * rsqr * std::log10(std::sqrt(rsqr));
            }

            S(row, col) = a(0) + a(1)*xi2 + a(2)*yi2 + sum;
        }
    }

    return S;
}

std::vector<PointId> MHCFilter::computeResiduals(Eigen::MatrixXd surface,
    PointViewPtr view, size_t rows, size_t cols, double res, BOX2D bounds,
    double tol)
{
    // for each point, compute diff with surface in 3x3 kernel window
    // if more than four diffs are within tolerance, add to ground
    
    std::vector<PointId> ground;
    
    for (PointId i = 0; i < view->size(); ++i)
    {
        using namespace Dimension;
        
        double x = view->getFieldAs<double>(Id::X, i);
        double y = view->getFieldAs<double>(Id::Y, i);
        double z = view->getFieldAs<double>(Id::Z, i);

        size_t c = floor(x-bounds.minx)/res;
        size_t r = floor(y-bounds.miny)/res;
        
        // for each cell in window
        int n(0);
        
        size_t cs = Utils::clamp(size_t(c-1), size_t(0), size_t(cols-1));
        size_t ce = Utils::clamp(size_t(c+1), size_t(0), size_t(cols-1));
        size_t rs = Utils::clamp(size_t(r-1), size_t(0), size_t(rows-1));
        size_t re = Utils::clamp(size_t(r+1), size_t(0), size_t(rows-1));
        
        // we can go out of range still
        for (size_t cc = cs; cc < ce; ++cc)
        {
            for (size_t rr = rs; rr < re; ++rr)
            {
                if (std::fabs(surface(rr, cc)-z) < tol)
                {
                    n++;
                    // std::cerr << rr << "\t" << cc << "\t" << surface(rr, cc) << "\t" << z << "\t" << tol << "\t" << n << std::endl;
                }
                if (n > 3)
                {
                    ground.push_back(i);
                    break; // remind myself, does this only break current loop?
                }
            }
            if (n > 3)
                break; // i think we have to exit both loops
        }
    }
    
    return ground;
}

std::vector<PointId> MHCFilter::processGround(PointViewPtr view)
{
    using namespace Eigen;
    
    log()->get(LogLevel::Info) << "Running MHC...\n";
    
    SpatialReference srs(view->spatialReference());
  
    // Calculate bounds, extents, number of columns and rows, etc.
    log()->get(LogLevel::Info) << "Calculating bounds...\n";
    BOX2D bounds;
    view->calculateBounds(bounds);
    size_t cols = ((bounds.maxx - bounds.minx) / m_res) + 1;
    size_t rows = ((bounds.maxy - bounds.miny) / m_res) + 1;
    log()->get(LogLevel::Debug) << "Rows: " << rows << "\t"
                                << "Cols: " << cols << std::endl;
    
    // Seed ground surface by finding coordinates of minimum elevations for each
    // grid cell at starting resolution, excluding low outliers.
    log()->get(LogLevel::Info) << "Computing extended local minimum...\n";
    std::vector<PointId> seeds = Utils::extendedLocalMinimum(*view.get(), rows,
        cols, m_res, bounds);
    log()->get(LogLevel::Debug) << "Generated " << seeds.size() << " seeds\n";

    for (int iter = 0; iter < 3; ++iter)
    {    
        cols = ((bounds.maxx - bounds.minx) / m_res) + 1;
        rows = ((bounds.maxy - bounds.miny) / m_res) + 1;
        log()->get(LogLevel::Debug) << "Rows: " << rows << "\t"
                                    << "Cols: " << cols << std::endl;
                                    
        point_count_t prevSeedCount(seeds.size());
          
        // infinite loop until we break
        while (true) {
            log()->get(LogLevel::Info) << "Creating seed PointView...\n";
            PointViewPtr seedView = view->makeNew();
            for (auto const& i : seeds)
                seedView->appendPoint(*view, i);
            
            // create raster using grid centers and TPS from the seed indices
            log()->get(LogLevel::Info) << "Reticulating splines...\n";
            MatrixXd surface = computeSpline(seedView, rows, cols, m_res, bounds);
            
            // log()->get(LogLevel::Info) << "Writing outputs...\n";
            // if (!m_outDir.empty())
            // {
            //     std::string filename = FileUtils::toAbsolutePath("surf.tif", m_outDir);
            //     eigen::writeMatrix(surface, filename, "GTiff", m_res, bounds, srs);
            // }
            
            // compute residuals
            
            // classify as ground if more than four residuals in 3x3 window are within
            // tolerance
            log()->get(LogLevel::Info) << "Analyzing residuals...\n";
            std::vector<PointId> ground = computeResiduals(surface, view, rows, cols,
                m_res, bounds, m_thresh);
            log()->get(LogLevel::Debug) << "Found " << ground.size() << " points near ground\n";
            
            // add ground points to seeds
            // seeds.insert(seeds.end(), ground.begin(), ground.end());
            std::set<PointId> idset(seeds.begin(), seeds.end());
            idset.insert(ground.begin(), ground.end());
            seeds.assign(idset.begin(), idset.end());
            // possibly duplicates? should we just use a set instead?
            
            log()->get(LogLevel::Debug) << "Ground + seeds: " << seeds.size()
                      << "\tPrevious: " << prevSeedCount << std::endl;
            
            // if ((after-before) == 0)
            if (seeds.size() == prevSeedCount)
                break;
            
            prevSeedCount = seeds.size();
            
            // local minimum at h to refine ground/seeds
            log()->get(LogLevel::Info) << "Computing local minimum...\n";
            std::vector<PointId> newseeds = Utils::localMinimum(*view.get(), seeds,
                rows, cols, m_res, bounds);
            log()->get(LogLevel::Debug) << "Seeds after local minimum " << newseeds.size() << std::endl;
            
            // repeat, etc., etc.
            // std::cerr << seeds.size() << "\t" << newseeds.size() << std::endl;
            seeds.swap(newseeds);
        }
        
        m_res *= 0.5;
        m_thresh += 0.1;
        log()->get(LogLevel::Info) << "Iteration " << iter << "\t"
                                   << "resolution " << m_res << "\t"
                                   << "tolerance " << m_thresh << std::endl;
    }
                                                 
    std::vector<PointId> groundIdx;
    groundIdx.swap(seeds);
    return groundIdx;
}

PointViewSet MHCFilter::run(PointViewPtr view)
{
    log()->get(LogLevel::Info) << "Process MHCFilter...\n";

    std::vector<PointId> idx = processGround(view);

    PointViewSet viewSet;

    if (!idx.empty() && (m_classify || m_extract))
    {

        if (m_classify)
        {
            log()->get(LogLevel::Debug) << "Labeled " << idx.size() << " ground returns!\n";

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
            log()->get(LogLevel::Debug) << "Extracted " << idx.size() << " ground returns!\n";

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
            log()->get(LogLevel::Debug) << "Filtered cloud has no ground returns!\n";

        if (!(m_classify || m_extract))
            log()->get(LogLevel::Debug) << "Must choose --classify or --extract\n";

        // return the view buffer unchanged
        viewSet.insert(view);
    }

    return viewSet;
}

} // namespace pdal
