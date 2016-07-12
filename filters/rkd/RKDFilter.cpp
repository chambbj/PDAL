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

void RKDFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;
    m_rangeDensity = layout->registerOrAssignDim("RangeDensity", Type::Double);
    m_frameNumber = layout->registerOrAssignDim("Frame Number", Type::Double);
    m_pixelNumber = layout->registerOrAssignDim("Pixel Number", Type::Double);
}

// void RKDFilter::prepared(PointTableRef table)
// {
    // const PointLayoutPtr layout(table.layout());
    // if (!layout->hasDim(Dimension::Id::FrameNumber))
        // throw pdal_error("RKDFilter: missing FrameNumber dimension in input PointView");
    // if (!layout->hasDim(Dimension::Id::PixelNumber))
        // throw pdal_error("RKDFilter: missing PixelNumber dimension in input PointView");
// }

void RKDFilter::filter(PointView& view)
{
    using namespace Eigen;
    using namespace Dimension;
  
    std::map<int, std::vector<PointId> > frameIdMap;
    
    double frame_min = std::numeric_limits<double>::max();
    double frame_max = std::numeric_limits<double>::lowest();
    for (PointId i = 0; i < view.size(); ++i)
    {
        double f = view.getFieldAs<double>(m_frameNumber, i);
        if (f < frame_min)
            frame_min = f;
        if (f > frame_max)
            frame_max = f;
            
        std::vector<PointId> ids = frameIdMap[f];
        ids.push_back(i);
        frameIdMap[f] = ids;
    }
    log()->get(LogLevel::Debug) << "Min frame = " << frame_min << std::endl;
    log()->get(LogLevel::Debug) << "Max frame = " << frame_max << std::endl;

    for (double f = frame_min; f < frame_max; ++f)
    {
        // log()->get(LogLevel::Debug) << "Processing frame " << f << std::endl;
    
        double sx, sy, sz;
        std::vector<PointId> ids = frameIdMap[f];
        std::vector<PointId> newIds;
        
        for (auto const& i : ids)
        {
            // get the sensor position (PixelNumber == -5) for this frame
            double p = view.getFieldAs<double>(m_pixelNumber, i);
            if (p == -5)
            {
                // record the sensor position
                sx = view.getFieldAs<double>(Id::X, i);
                sy = view.getFieldAs<double>(Id::Y, i);
                sz = view.getFieldAs<double>(Id::Z, i);
                // log()->get(LogLevel::Debug) << "Found sensor at " << sx << "\t" << sy << "\t" << sz << std::endl;;
                continue;
            }
                
            if (p < 0)
                continue;
                
            newIds.push_back(i);
        }
        // log()->get(LogLevel::Debug) << "Frame has " << newIds.size() << " points\n";
        
        if (newIds.size() == 0)
            continue;
            
        VectorXd range(newIds.size());
        range.setZero();
        
        for (size_t i = 0; i < newIds.size(); ++i)
        {
            double x = view.getFieldAs<double>(Id::X, newIds[i]);
            double y = view.getFieldAs<double>(Id::Y, newIds[i]);
            double z = view.getFieldAs<double>(Id::Z, newIds[i]);
            
            range(i) = std::sqrt((x-sx)*(x-sx)+(y-sy)*(y-sy)+(z-sz)*(z-sz));
            
        }
        
        VectorXd density(newIds.size());
        density.setZero();
        std::multiset<double> sorted_density;
        
        // compute range density for the frame
        for (size_t i = 0; i < newIds.size(); ++i)
        {
            // create a copy of the vector with current range removed
            VectorXd subset = range;
            subset.segment(i, range.size()-i-1) = subset.segment(i+1, range.size()-i-1);
            subset.conservativeResize(range.size()-1);
            
            subset = subset - VectorXd::Constant(range.size()-1, range(i));
            subset /= 0.5;
            subset = subset.cwiseProduct(subset);
            subset *= -0.5;
            subset = subset.array().exp().matrix();  // legit?
            subset /= std::sqrt(2*3.14159);
            density(i) = subset.sum() / (subset.size()*0.5);
            sorted_density.insert(density(i));
        }
        
        // log()->get(LogLevel::Debug) << density.transpose() << std::endl;
        // density.normalize();
        // log()->get(LogLevel::Debug) << density.transpose() << std::endl;
        
        int thresh_id = static_cast<int>(std::ceil(0.9 * density.size()));
        double thresh;
        int j = 0;
        for (auto it = sorted_density.begin(); it != sorted_density.end(); ++it)
        {
            if (j++ < thresh_id)
                continue;
            thresh = *it;
            break;
        }
        log()->get(LogLevel::Debug) << thresh << std::endl;
          
        for (size_t i = 0; i < density.size(); ++i)
        {
            if (density(i) > thresh)
                view.setField(m_rangeDensity, newIds[i], 1.0);
            else
                view.setField(m_rangeDensity, newIds[i], 0.0);
        }
    }
}


} // namespace pdal
