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
    // m_rangeDensity = layout->registerOrAssignDim("Density", Type::Double);
    layout->registerDim(Id::Intensity);
    layout->registerDim(Id::Reflectance);
    layout->registerDim(Id::ReturnNumber);
    layout->registerDim(Id::NumberOfReturns);
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

    // All my vectors are out here b/c I thought there may be some savings in just allocating them once, and resizing, resetting as needed in the loops. Not sure it made that much of a difference...
    VectorXd MAPCPNeighbors = VectorXd::Zero(n);
    VectorXd density = VectorXd::Zero(n);
    VectorXd x_vals, y_vals, z_vals;
    VectorXd x_diff, y_diff, z_diff, temp;
    // VectorXd vals = VectorXd::Zero(n-2);
    VectorXd peaks = VectorXd::Zero(n-2);
    VectorXd area = VectorXd::Zero(n-2);
    VectorXd areaFrac = VectorXd::Zero(n-2);
    VectorXd diff = VectorXd::Zero(n-1);
    VectorXd sign = VectorXd::Zero(n-1);
    VectorXd diff2 = VectorXd::Zero(n-2);

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

            // Record values from each of the neighbors.
            x_vals.resize(neighbors.size());
            y_vals.resize(neighbors.size());
            z_vals.resize(neighbors.size());
            for (PointId idx = 0; idx < neighbors.size(); ++idx)
            {
                x_vals(idx) = view->getFieldAs<double>(Id::X, neighbors[idx]);
                y_vals(idx) = view->getFieldAs<double>(Id::Y, neighbors[idx]);
                z_vals(idx) = view->getFieldAs<double>(Id::Z, neighbors[idx]);
            }

            x_diff.resize(neighbors.size());
            y_diff.resize(neighbors.size());
            z_diff.resize(neighbors.size());
            temp.resize(neighbors.size());

            // Sample density for the current column.
            double invbw = 1 / m_bw;
            double invdenom = 1 / std::sqrt(2*3.14159);
            double invdenom2 = 1 / (temp.size() * m_bw);
            for (size_t i = 0; i < samples.size(); ++i)
            {
                x_diff = x_vals - VectorXd::Constant(x_vals.size(), x);
                y_diff = y_vals - VectorXd::Constant(y_vals.size(), y);
                z_diff = z_vals - VectorXd::Constant(z_vals.size(), samples(i));
                x_diff = x_diff.cwiseProduct(x_diff);
                y_diff = y_diff.cwiseProduct(y_diff);
                z_diff = z_diff.cwiseProduct(z_diff);
                temp = (x_diff + y_diff + z_diff).cwiseSqrt() * invbw;
                temp = temp.cwiseProduct(temp) * -0.5;
                temp = temp.array().exp().matrix() * invdenom;
                density(i) = temp.sum() * invdenom2;
            }
            // how critical is it to normalize this in some way? it does affect the peak area, the overall area, and therefor the intensity and reflectance
            // density /= density.maxCoeff();

            // std::cerr << "density\n";
            // std::cerr << density.transpose() << std::endl;

            auto diffEq = [](VectorXd vec)
            {
                return vec.tail(vec.size()-1)-vec.head(vec.size()-1);
            };

            // MATLAB diff command - approximate derivative
            diff = diffEq(density);

            // MATLAB sign command - sigmoid function
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
            diff2 = diffEq(sign);

            // std::cerr << "diff2\n";
            // std::cerr << diff2.transpose() << std::endl;

            // vals.resize(n-2);
            peaks.resize(n-2);
            area.resize(n-2);
            areaFrac.resize(n-2);
            
            double invdensitysum = 1 / density.sum();

            // Peaks occur at diff2 == -2
            int nPeaks = 0;
            int nrad = 3;
            double rad = (2*nrad+1)*m_hres/2;
            for (int i = 0; i < samples.size()-2; ++i)
            {
                if (diff2(i) == -2)
                {
                    int nei = kd3.radius(x, y, samples(i), rad).size();
                    if (nei < 4)
                        continue;
                        
                    double peakArea = density(i);
                    for (int j = i+1; j < samples.size()-2; ++j)
                    {
                        if (diff2(j) > 0)
                            break;
                        peakArea += density(j);
                    }
                    for (int j = i-1; j >= 0; --j)
                    {
                        if (diff2(j) > 0)
                            break;
                        peakArea += density(j);
                    }
                    // if (peakArea * invdensitysum < 0.1)
                    //     continue;

                    // vals(nPeaks) = density(i);
                    peaks(nPeaks) = samples(i);
                    area(nPeaks) = peakArea;
                    areaFrac(nPeaks) = peakArea * invdensitysum;

                    nPeaks++;
                }
            }

            if (nPeaks == 0)
                continue;

            // vals.conservativeResize(nPeaks);
            peaks.conservativeResize(nPeaks);
            area.conservativeResize(nPeaks);
            areaFrac.conservativeResize(nPeaks);

            // vals /= vals.sum();

            // For each peak of sufficient size/strength, find the nearest
            // neighbor in the raw data and append to the output view.
            for (int i = 0; i < nPeaks; ++i)
            {
                PointIdVec idx = kd3.neighbors(x, y, peaks(i), 1);
                // view->setField(m_rangeDensity, idx[0], vals(i));
                view->setField(Id::NumberOfReturns, idx[0], nPeaks);
                view->setField(Id::ReturnNumber, idx[0], nPeaks-i);
                view->setField(Id::Intensity, idx[0], area(i));
                view->setField(Id::Reflectance, idx[0], areaFrac(i));
                output->appendPoint(*view, idx[0]);
            }
        }
    }

    viewSet.erase(view);
    viewSet.insert(output);
    return viewSet;
}

} // namespace pdal
