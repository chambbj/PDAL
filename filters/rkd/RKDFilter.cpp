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

#include <random>
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
    args.add("nsamps", "Number of samples", m_nSamples, 256);
    args.add("bw", "Bandwidth", m_bw, 0.15);
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
    
    PointViewSet viewSet;
    
    // compute bounds for zmin, zmax
    BOX3D bounds;
    view->calculateBounds(bounds);
    
    double x = (bounds.maxx - bounds.minx) / 2 + bounds.minx;
    double y = (bounds.maxy - bounds.miny) / 2 + bounds.miny;
    
    // sampled uniformly between zmin and zmax;
    m_samples.resize(m_nSamples);
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(bounds.minz, bounds.maxz);
    double spacing = (bounds.maxz - bounds.minz) / m_nSamples;
    for (int n = 0; n < m_nSamples; ++n)
        m_samples(n) = bounds.minz + n*spacing;
        // m_samples(n) = dis(gen);
    // log()->get(LogLevel::Debug) << "initialize() should only be called once\n";
    // log()->get(LogLevel::Debug) << m_samples.transpose() << std::endl;
    
    // log()->get(LogLevel::Debug) << "Processing view with " << view->size() << " points\n";
    m_measurements.resize(view->size());
  
    PointRef point(*view, 0);
    for (PointId idx = 0; idx < view->size(); ++idx)
    {
        point.setPointId(idx);
        m_measurements(idx) = point.getFieldAs<double>(Dimension::Id::Z);
    }
    
    // log()->get(LogLevel::Debug) << "Found " << numGood << " measurements\n";
    // m_measurements.conservativeResize(numGood);
    
    // log()->get(LogLevel::Debug) << "Initial samples\n";
    // log()->get(LogLevel::Debug) << m_samples.transpose() << std::endl;
    
    // log()->get(LogLevel::Debug) << "measurements\n";
    // log()->get(LogLevel::Debug) << m_measurements.transpose() << std::endl;
    
    VectorXd density(m_nSamples);
    density.setZero();
    
    // compute range density for the frame
    for (size_t i = 0; i < m_nSamples; ++i)
    {
        // create a copy of the vector with current range removed
        VectorXd subset = m_measurements;
        int N = m_measurements.size();
        
        subset = subset - VectorXd::Constant(N, m_samples(i));
        subset /= m_bw;
        subset = subset.cwiseProduct(subset);
        subset *= -0.5;
        subset = subset.array().exp().matrix();  // legit?
        subset /= std::sqrt(2*3.14159);
        density(i) = subset.sum() / (subset.size()*m_bw);
    }
    density /= density.maxCoeff();
    
    // log()->get(LogLevel::Debug) << "densities\n";
    // log()->get(LogLevel::Debug) << density << std::endl;
    
    VectorXd diff(m_nSamples-1);
    for (int i = 1; i < m_nSamples; ++i)
        diff(i-1) = density(i) - density(i-1);
    
    VectorXd sign(m_nSamples-1);
    for (int i = 0; i < m_nSamples-1; ++i)
    {
        if (diff(i) < 0)
            sign(i) = -1;
        else if (diff(i) > 0)
            sign(i) = 1;
        else
            sign(i) = 0;
    }
    
    VectorXd diff2(m_nSamples-2);
    for (int i = 1; i < m_nSamples-1; ++i)
        diff2(i-1) = sign(i) - sign(i-1);
    
    VectorXd vals(m_nSamples-2);
    vals.setZero();
    VectorXd newSamples(m_nSamples-2);
    newSamples.setZero();
    int nPeaks = 0;    
    for (int i = 0; i < m_nSamples-2; ++i)
    {
        if (diff2(i) == -2)
        {
            vals(nPeaks) = density(i);
            newSamples(nPeaks++) = m_samples(i);
        }
    }
    
    if (nPeaks == 0)
        return viewSet;
    
    vals.conservativeResize(nPeaks);
    vals /= vals.maxCoeff();
    double minval = vals.minCoeff();
    newSamples.conservativeResize(nPeaks);
    // log()->get(LogLevel::Debug) << vals.minCoeff() << "\t" << vals.maxCoeff() << "\t" << nPeaks << std::endl;
    
    PointViewPtr output = view->makeNew();
    PointId idx = 0;
    for (int i = 0; i < nPeaks; ++i)
    {
        // create a point at x, y, m_samples(i)
        // if (diff2(i) == -2)
        if (vals(i) == 1.0)
        {
            // log()->get(LogLevel::Debug) << m_samples(i) << "\t" << density(i) << "\t" << diff(i) << "\t" << sign(i) << "\t" << diff2(i) << std::endl;
            
            // output->appendPoint(*view, i);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> disx(bounds.minx, bounds.maxx);
            std::uniform_real_distribution<> disy(bounds.miny, bounds.maxy);
            output->setField(Dimension::Id::X, idx, x /*disx(gen)*/);
            output->setField(Dimension::Id::Y, idx, y /*disy(gen)*/);
            output->setField(Dimension::Id::Z, idx, newSamples(i));
            output->setField(m_rangeDensity, idx, vals(i));
            idx++;
        }
    }
    
    viewSet.erase(view);
    viewSet.insert(output);
    
    // for (size_t i = 0; i < density.size(); ++i)
    // {
    //     log()->get(LogLevel::Debug) << density(i) << std::endl;
    //     // if (density(i) > thresh)
    //     if (std::isnan(density(i)) || std::isinf(density(i)))
    //         continue;
    //         // view.setField(m_rangeDensity, newIds[i], 0.0);
    //     view.setField(m_rangeDensity, newIds[i], density(i));
    //     // else
    //     //     view.setField(m_rangeDensity, newIds[i], 0.0);
    // }
    // 
    // if (std::isnan(density.sum()))
    //     return viewSet;
    
    // log()->get(LogLevel::Debug) << "Weights\n";
    // log()->get(LogLevel::Debug) << density.transpose() << std::endl;
    // log()->get(LogLevel::Debug) << "Checksum " << density.sum() << std::endl;
    
    // viewSet.insert(m_view);
    return viewSet;
}

} // namespace pdal
