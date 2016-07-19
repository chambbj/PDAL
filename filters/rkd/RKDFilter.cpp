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
    m_rangeDensity = layout->registerOrAssignDim("Density", Type::Double);
}

void RKDFilter::filter(PointView& view)
{
    using namespace Eigen;
    using namespace Dimension;
    
        VectorXd range(newIds.size());
        range.setZero();
        
        for (size_t i = 0; i < newIds.size(); ++i)
        {
            double z = view.getFieldAs<double>(Id::Z, newIds[i]);
            range(i) = z;
        }
        
        VectorXd density(newIds.size());
        density.setZero();
        
        // compute range density for the frame
        for (size_t i = 0; i < newIds.size(); ++i)
        {
            double bw = 0.15;
            
            // create a copy of the vector with current range removed
            VectorXd subset = range;
            int N = range.size()-1;
            subset.segment(i, N-i) = subset.segment(i+1, N-i);
            subset.conservativeResize(N);
            
            subset = subset - VectorXd::Constant(N, range(i));
            // log()->get(LogLevel::Debug) << subset.transpose() << std::endl;
            subset /= bw;
            subset = subset.cwiseProduct(subset);
            subset *= -0.5;
            subset = subset.array().exp().matrix();  // legit?
            subset /= std::sqrt(2*3.14159);
            density(i) = subset.sum() / (subset.size()*bw);
        }
        
      density /= density.sum();
        
        for (size_t i = 0; i < density.size(); ++i)
        {
            log()->get(LogLevel::Debug) << density(i) << std::endl;
            // if (density(i) > thresh)
            if (std::isnan(density(i)) || std::isinf(density(i)))
                continue;
                // view.setField(m_rangeDensity, newIds[i], 0.0);
            view.setField(m_rangeDensity, newIds[i], density(i));
            // else
            //     view.setField(m_rangeDensity, newIds[i], 0.0);
        }
}


} // namespace pdal
