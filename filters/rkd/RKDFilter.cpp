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

struct peak
{
    double loc;
    double area;
};

struct by_area
{
    bool operator()(peak const& a, peak const& b)
    {
        return a.area > b.area;
    }
};

Eigen::VectorXf RKDFilter::diffEq(Eigen::VectorXf const& vec)
{
    return vec.tail(vec.size()-1)-vec.head(vec.size()-1);
}

std::string RKDFilter::getName() const
{
    return s_info.name;
}

void RKDFilter::addArgs(ProgramArgs& args)
{
    // args.add("bw", "Bandwidth", m_bw, 0.6);
    args.add("hres", "Horizontal resolution", m_hres, 0.05);
    args.add("vres", "Vertical resolution", m_vres, 0.05);
    // args.add("radius", "Radius", m_radius, 0.15);
}

void RKDFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;
    layout->registerDim(Id::Amplitude);
}

PointViewSet RKDFilter::run(PointViewPtr view)
{
    using namespace Eigen;
    using namespace Dimension;

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

    // Considering not having this be a user-defined parameter, rather hold it
    // constant at 1.5 (equivalent to 1 bin in X and Y).
    m_radius = 1.5 * m_hres;

    VectorXf density = VectorXf::Zero(n);
    ArrayXf z_diff, xyprod;

    // Initialize the samples.
    VectorXf samples(n);
    for (auto i = 0; i < samples.size(); ++i)
        samples(i) = i * m_vres;

    for (auto r = 0; r < rows; ++r)
    {
        float y = r * m_hres;
        for (auto c = 0; c < cols; ++c)
        {
            float x = c * m_hres;

            // Find neighbors in raw cloud at current XY cell.
            std::vector<PointId> neighbors = kd2.radius(x + bounds.minx,
                                             y + bounds.miny,
                                             m_radius);

            // Record values from each of the neighbors.
            ArrayXf x_vals(neighbors.size());
            ArrayXf y_vals(neighbors.size());
            ArrayXf z_vals(neighbors.size());

            for (PointId idx = 0; idx < neighbors.size(); ++idx)
            {
                x_vals(idx) = view->getFieldAs<double>(Id::X, neighbors[idx]) - bounds.minx;
                y_vals(idx) = view->getFieldAs<double>(Id::Y, neighbors[idx]) - bounds.miny;
                z_vals(idx) = view->getFieldAs<double>(Id::Z, neighbors[idx]) - bounds.minz;
            }

            auto setBandwidth = [](float std, int n)
            {
                return std * 2.34f * std::pow(n, -0.2f);
            };

            float h_x = setBandwidth(0.15f, neighbors.size());
            float h_y = setBandwidth(0.15f, neighbors.size());
            float h_z = setBandwidth(0.30f, neighbors.size());

            // Sample density for the current column.
            float factor = 0.75f / (neighbors.size() * h_x * h_y * h_z);

            auto applyK = [](float x)
            {
                if (std::abs(x) > 1)
                    return 0.0f;
                else
                    return 1.0f - x * x;
            };

            xyprod = ((x_vals - x) / h_x).unaryExpr(std::ref(applyK)) *
                     ((y_vals - y) / h_y).unaryExpr(std::ref(applyK));

            for (auto i = 0; i < samples.size(); ++i)
            {
                z_diff = (z_vals - samples(i)) / h_z;
                density(i) = factor * (xyprod * (z_diff).unaryExpr([](float x)
                {
                    if (std::abs(x) > 1)
                        return 0.0f;
                    else
                        return 1.0f - x * x;
                })).sum();
            }
            density /= density.sum();

            // auto diffEq = [](VectorXf const& vec)
            // {
            //     return vec.tail(vec.size()-1)-vec.head(vec.size()-1);
            // };

            // MATLAB diff command - approximate derivative
            VectorXf diff = diffEq(density);

            // MATLAB sign function
            auto signFcn = [](float x)
            {
                if (x < 0.0f)
                    return -1.0f;
                else if (x > 0.0f)
                    return 1.0f;
                else
                    return 0.0f;
            };
            VectorXf sign = diff.unaryExpr(std::ref(signFcn));

            // MATLAB diff command again - approxiate derivative
            VectorXf diff2 = diffEq(sign);

            // Peaks occur at diff2 == -2
            int nPeaks = 0;
            std::vector<peak> pvec;
            for (auto i = 0; i < diff2.size(); ++i)
            {
                if (diff2(i) == -2)
                {
                    double peakArea = density(i);
                    for (int j = i+1; j < diff2.size(); ++j)
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
                    pvec.push_back(peak{samples(i), peakArea});

                    nPeaks++;
                }
            }

            std::sort(pvec.begin(), pvec.end(), by_area());

            if (nPeaks == 0)
                continue;

            // For each peak of sufficient size/strength, find the nearest
            // neighbor in the raw data and append to the output view.
            for (auto const& p : pvec)
            {
                if (p.area < 0.1)
                    break;

                std::vector<PointId> idx = kd3.neighbors(x + bounds.minx,
                                           y + bounds.miny,
                                           p.loc + bounds.minz,
                                           1);
                view->setField(Id::Amplitude, idx[0], p.area);
                output->appendPoint(*view, idx[0]);
            }
        }
    }

    viewSet.erase(view);
    viewSet.insert(output);
    return viewSet;
}

} // namespace pdal
