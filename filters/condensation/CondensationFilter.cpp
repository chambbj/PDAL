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

#include "CondensationFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>

#include <Eigen/Dense>

#include <random>
#include <string>
#include <vector>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.condensation", "Condensation Filter",
               "http://pdal.io/stages/filters.condensation.html");

CREATE_STATIC_PLUGIN(1, 0, CondensationFilter, Filter, s_info)

std::string CondensationFilter::getName() const
{
    return s_info.name;
}

void CondensationFilter::addArgs(ProgramArgs& args)
{
    args.add("xmin", "Minimum X", m_xmin);
    args.add("xmax", "Maximum X", m_xmax);
    args.add("ymin", "Minimum Y", m_ymin);
    args.add("ymax", "Maximum Y", m_ymax);
    args.add("zmin", "Minimum Z", m_zmin);
    args.add("zmax", "Maximum Z", m_zmax);
    args.add("nsamps", "Number of samples", m_nSamples);
}

void CondensationFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;
    m_pixelNumber = layout->registerOrAssignDim("Pixel Number", Type::Double);
}

void CondensationFilter::initialize()
{
    // initialize samples, based on options, esp. bbox
    // sampled uniformly between zmin and zmax;
    m_samples.resize(m_nSamples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(m_zmin, m_zmax);
    for (int n = 0; n < m_nSamples; ++n)
        m_samples(n) = dis(gen);
    log()->get(LogLevel::Debug) << "initialize() should only be called once\n";
    log()->get(LogLevel::Debug) << m_samples.transpose() << std::endl;
}

void CondensationFilter::ready(PointTableRef table)
{
    SpatialReference srs = getSpatialReference();

    if (srs.empty())
        srs = table.anySpatialReference();
    m_view.reset(new PointView(table, srs));
}

Eigen::VectorXd CondensationFilter::computeDensity()
{
    using namespace Eigen;
    using namespace Dimension;

    VectorXd density(m_nSamples);
    density.setZero();
    
    // compute range density for the frame
    for (size_t i = 0; i < m_nSamples; ++i)
    {
        double bw = 0.15;
        
        // create a copy of the vector with current range removed
        VectorXd subset = m_measurements;
        int N = m_measurements.size();
        
        subset = subset - VectorXd::Constant(N, m_samples(i));
        subset /= bw;
        subset = subset.cwiseProduct(subset);
        subset *= -0.5;
        subset = subset.array().exp().matrix();  // legit?
        subset /= std::sqrt(2*3.14159);
        density(i) = subset.sum() / (subset.size()*bw);
    }
    density /= density.sum();
    return density;
}

bool CondensationFilter::processOne(PointRef& point)
{
    double x = point.getFieldAs<double>(Dimension::Id::X);
    double y = point.getFieldAs<double>(Dimension::Id::Y);
    double z = point.getFieldAs<double>(Dimension::Id::Z);
    double p = point.getFieldAs<double>(m_pixelNumber);
    if (x < m_xmin || x > m_xmax)
        return false;
    if (y < m_ymin || y > m_ymax)
        return false;
    if (z < m_zmin || z > m_zmax)
        return false;
    if (p < 0)
        return false;
    
    return true;
}

PointViewSet CondensationFilter::run(PointViewPtr view)
{
    using namespace Eigen;
    
    PointViewSet viewSet;

    // If the SRS of all the point views aren't the same, print a warning
    // unless we're explicitly overriding the SRS.
    if (getSpatialReference().empty() &&
      (view->spatialReference() != m_view->spatialReference()))
        log()->get(LogLevel::Warning) << getName() << ": merging points "
            "with inconsistent spatial references." << std::endl;
    
    log()->get(LogLevel::Debug) << "Processing view with " << view->size() << " points\n";
    m_measurements.resize(view->size());
  
    PointRef point(*view, 0);
    int numGood = 0;
    for (PointId idx = 0; idx < view->size(); ++idx)
    {
        point.setPointId(idx);
        if (processOne(point))
            m_measurements(numGood++) = point.getFieldAs<double>(Dimension::Id::Z);
    }
    
    if (numGood == 0)
        return viewSet;
    
    // log()->get(LogLevel::Debug) << "Found " << numGood << " measurements\n";
    m_measurements.conservativeResize(numGood);
    
    // log()->get(LogLevel::Debug) << "Initial samples\n";
    // log()->get(LogLevel::Debug) << m_samples.transpose() << std::endl;
    
    log()->get(LogLevel::Debug) << "measurements\n";
    log()->get(LogLevel::Debug) << m_measurements.transpose() << std::endl;
    
    // then compute density with samples and measurements
    VectorXd density = computeDensity();
    
    if (std::isnan(density.sum()))
        return viewSet;
    
    // log()->get(LogLevel::Debug) << "Weights\n";
    // log()->get(LogLevel::Debug) << density.transpose() << std::endl;
    // log()->get(LogLevel::Debug) << "Checksum " << density.sum() << std::endl;
    
    VectorXd c(density.size());
    c(0) = 0.0;
    for (int i = 1; i < density.size(); ++i)
    {
        c(i) = c(i-1) + density(i);
    }
    
    // log()->get(LogLevel::Debug) << "Cumulative weights\n";
    // log()->get(LogLevel::Debug) << c.transpose() << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    VectorXd new_samples(m_nSamples);
    for (int n = 0; n < m_nSamples; ++n)
    {
        double v = dis(gen);
        for (int i = 0; i < c.size(); ++i)
        {
            if (c(i) > v)
            {
              // log()->get(LogLevel::Debug) << c(i) << " > " << v << " at " << m_samples(i) << " with weight " << density(i) << std::endl;
              new_samples(n) = m_samples(i);
              break;
            }
        }
    }
    m_samples.swap(new_samples);
    
    std::normal_distribution<> dis2(0, 0.15);
    for (int n = 0; n < m_nSamples; ++n)
    {
        m_samples(n) += dis2(gen);
    }
    
    log()->get(LogLevel::Debug) << "Diffused samples\n";
    log()->get(LogLevel::Debug) << m_samples.transpose() << std::endl;
    
    // viewSet.insert(m_view);
    return viewSet;
}

} // namespace pdal
