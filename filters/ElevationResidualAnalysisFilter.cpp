/******************************************************************************
 * Copyright (c) 2017, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "ElevationResidualAnalysisFilter.hpp"

#include <pdal/KDIndex.hpp>

#include <string>
#include <vector>

namespace pdal
{

static PluginInfo const s_info
{
    "filters.era",
    "ElevationResidualAnalysis Filter",
    "http://pdal.io/stages/filters.era.html"
};

CREATE_STATIC_STAGE(ElevationResidualAnalysisFilter, s_info)

std::string ElevationResidualAnalysisFilter::getName() const
{
    return s_info.name;
}

ElevationResidualAnalysisFilter::ElevationResidualAnalysisFilter() : Filter() {}

void ElevationResidualAnalysisFilter::addDimensions(PointLayoutPtr layout)
{
    using namespace Dimension;

    m_mean = layout->registerOrAssignDim("MeanElevation", Type::Double);
    m_diffmean = layout->registerOrAssignDim("DiffMeanElevation", Type::Double);
    m_range = layout->registerOrAssignDim("Range", Type::Double);
    m_std = layout->registerOrAssignDim("StdevElevation", Type::Double);
    m_devmean = layout->registerOrAssignDim("DevMeanElevation", Type::Double);
}

void ElevationResidualAnalysisFilter::filter(PointView& view)
{
    KD2Index& kdi = view.build2dIndex();

    for (PointRef p : view)
    {
        PointIdList neighbors = kdi.radius(p, std::sqrt(2.0));
        double val(0.0);
        double maxZ(std::numeric_limits<double>::lowest());
        double minZ(std::numeric_limits<double>::max());
        for (PointId const& n : neighbors)
        {
            PointRef point = view.point(n);
            double z(point.getFieldAs<double>(Dimension::Id::Z));
            val += z;
            if (z > maxZ)
                maxZ = z;
            if (z < minZ)
                minZ = z;
        }
        val /= neighbors.size();
        double range(maxZ - minZ);

        double std(0.0);
        for (PointId const& n : neighbors)
        {
            PointRef point = view.point(n);
            double z(point.getFieldAs<double>(Dimension::Id::Z));
            std += (z - val) * (z - val);
        }
        std /= (neighbors.size() - 1);
        double stdev(std::sqrt(std));

        double z(p.getFieldAs<double>(Dimension::Id::Z));
        p.setField(m_mean, val);
        p.setField(m_diffmean, z - val);
        p.setField(m_range, range);
        p.setField(m_std, stdev);
        p.setField(m_devmean, (z - val) / stdev);
    }
}

} // namespace pdal
