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

#include "RKDFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.rkd", "RKD Filter",
               "http://pdal.io/stages/filters.rkd.html");

CREATE_STATIC_PLUGIN(1, 0, RKDFilter, Filter, s_info)

std::string RKDFilter::getName() const
{
    return s_info.name;
}

void RKDFilter::addArgs(ProgramArgs& args)
{
    args.add("bw", "Bandwidth", m_bw, 0.6);
    args.add("hres", "Horizontal resolution", m_hres, 0.05);
    args.add("vres", "Vertical resolution", m_vres, 0.05);
    args.add("radius", "Radius", m_radius, 0.15);
}

void RKDFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;
    m_rangeDensity = layout->registerOrAssignDim("Density", Type::Double);
}

PointViewSet RKDFilter::run(PointViewPtr view)
{
    using namespace Eigen;
    using namespace Dimension;
    
    typedef std::vector<PointId> PointIdVec;

    // The 2D index is used to find neighbors in a Z column.    
    KD2Index kd2(*view);
    kd2.build();
    
    // The 3D index is used to find the nearest neighbor.
    KD3Index kd3(*view);
    kd3.build();
    
    PointViewSet viewSet;
    
    // Create the output PointView.
    PointViewPtr output = view->makeNew();
    
    // Compute bounds to derive number of rows, cols, and samples.
    BOX3D bounds;
    view->calculateBounds(bounds);
    double extentx = bounds.maxx - bounds.minx;
    double extenty = bounds.maxy - bounds.miny;
    double extentz = bounds.maxz - bounds.minz;
    int cols = static_cast<int>(std::ceil(extentx / m_hres));
    int rows = static_cast<int>(std::ceil(extenty / m_hres));
    int n = static_cast<int>(std::ceil(extentz / m_vres));
    log()->get(LogLevel::Debug) << "# samples = " << n << std::endl;
    log()->get(LogLevel::Debug) << "# rows = " << rows << std::endl;
    log()->get(LogLevel::Debug) << "# cols = " << cols << std::endl;
  
    // Initialize the samples.
    VectorXd samples(n);
    for (int i = 0; i < samples.size(); ++i)
        samples(i) = bounds.minz + i * m_vres;
    
    for (int r = 0; r < rows; ++r)
    {
        double y = bounds.miny + r * m_hres;
        for (int c = 0; c < cols; ++c)
        {
            double x = bounds.minx + c * m_hres;
            
            // Find neighbors in raw cloud at current XY cell.
            PointIdVec neighbors = kd2.radius(x, y, m_radius);
            
            // Record Z values from each of the neighbors.
            VectorXd z_vals(neighbors.size());
            for (PointId idx = 0; idx < neighbors.size(); ++idx)
                z_vals(idx) = view->getFieldAs<double>(Id::Z, neighbors[idx]);
            // std::cerr << z_vals.transpose() << std::endl;
            
            // Sample density for the current column.
            VectorXd density = VectorXd::Zero(samples.size());
            for (size_t i = 0; i < samples.size(); ++i)
            {
                VectorXd temp = z_vals;
                temp = temp - VectorXd::Constant(temp.size(), samples(i));
                temp /= m_bw;
                temp = temp.cwiseProduct(temp);
                temp *= -0.5;
                temp = temp.array().exp().matrix();
                temp /= std::sqrt(2*3.14159);
                density(i) = temp.sum() / (temp.size()*m_bw);
            }
            density /= density.maxCoeff();
            // std::cerr << density.transpose() << std::endl;
            
            // MATLAB diff command - approximate derivative
            VectorXd diff(samples.size()-1);
            for (int i = 1; i < samples.size(); ++i)
                diff(i-1) = density(i) - density(i-1);
            
            // MATLAB sign command - sigmoid function
            VectorXd sign(samples.size()-1);
            for (int i = 0; i < samples.size()-1; ++i)
            {
                if (diff(i) < 0)
                    sign(i) = -1;
                else if (diff(i) > 0)
                    sign(i) = 1;
                else
                    sign(i) = 0;
            }
            
            // MATLAB diff command again - approxiate derivative
            VectorXd diff2(samples.size()-2);
            for (int i = 1; i < samples.size()-1; ++i)
                diff2(i-1) = sign(i) - sign(i-1);
            
            // Peaks occur at diff2 == -2
            VectorXd vals = VectorXd::Zero(samples.size()-2);
            VectorXd peaks = VectorXd::Zero(samples.size()-2);
            int nPeaks = 0;
            // std::cerr << diff2.transpose() << std::endl;
            for (int i = 0; i < samples.size()-2; ++i)
            {
                if (diff2(i) == -2)
                {
                    vals(nPeaks) = density(i);
                    peaks(nPeaks++) = samples(i);
                }
            }
            
            // printf("npeaks %d\n", nPeaks);
            
            if (nPeaks == 0)
                continue;
            
            vals.conservativeResize(nPeaks);
            peaks.conservativeResize(nPeaks);
            
            vals /= vals.maxCoeff();
            
            // Make a copy of the vals, to sort in place, and determine a threshold.
            // auto valcopy = vals;
            // std::sort(valcopy.data(), valcopy.data()+valcopy.size(),std::greater<double>());
            
            // Struggling with a good way to set a threshold. Highest peak is
            // pretty good. More would be better. Where do we stop?
            // double thresh;
            // if (nPeaks > 1)
            //     thresh = valcopy(1);
            // else
            //     thresh = valcopy(0);
            
            // For each peak of sufficient size/strength, find the nearest
            // neighbor in the raw data and append to the output view.
            for (int i = 0; i < nPeaks; ++i)
            {
              // printf("%0.2f\n", vals(i));
                // if (vals(i) > thresh)
                if (vals(i) == 1.0)
                {
                  // printf("Peak %d of %d at %0.2f (%0.2f > %0.2f)\n", i, nPeaks, peaks(i), vals(i), thresh);
                    PointIdVec idx = kd3.neighbors(x, y, peaks(i), 1);
                    output->appendPoint(*view, idx[0]);
                }
            }
        }
    }
    
    viewSet.erase(view);
    viewSet.insert(output);
    return viewSet;
}

} // namespace pdal
