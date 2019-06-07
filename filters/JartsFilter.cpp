/******************************************************************************
 * Copyright (c) 2019, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "JartsFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/util/ProgramArgs.hpp>
#include <pdal/util/Utils.hpp>

#include <string>
#include <vector>

namespace pdal
{

static StaticPluginInfo const s_info
{
    "filters.jarts",
    "Jarts filter",
    "http://pdal.io/stages/filters.jarts.html"
};

CREATE_STATIC_STAGE(JartsFilter, s_info)

std::string JartsFilter::getName() const
{
    return s_info.name;
}


void JartsFilter::addArgs(ProgramArgs& args)
{
    args.add("mr", "Masking radius", m_maskingRadius, 1.0);
    args.add("er", "Elevation radius", m_elevationRadius, 2.0);
    args.add("et", "Elevation threshold", m_elevationThreshold, 0.5);
}


PointViewSet JartsFilter::run(PointViewPtr inView)
{
    point_count_t np = inView->size();

    // Return empty PointViewSet if the input PointView has no points.
    // Otherwise, make a new output PointView.
    PointViewSet viewSet;
    if (!np)
        return viewSet;
    PointViewPtr outView = inView->makeNew();

    // Build the 2D KD-tree.
    KD2Index& index = inView->build2dIndex();

    // Find the minimum Z as a starting point for Jarts.
    double minZ = std::numeric_limits<double>::max();
    PointId idxZ;
    for (PointId i = 0; i < np; ++i)
    {
        double z = inView->getFieldAs<double>(Dimension::Id::Z, i);
        if (z < minZ)
        {
            minZ = z;
            idxZ = i;
        }
    }

    auto master_ids = index.neighbors(idxZ, np);
    log()->get(LogLevel::Debug) << idxZ << std::endl;
    log()->get(LogLevel::Debug) << master_ids.size() << std::endl;
    log()->get(LogLevel::Debug) << master_ids[0] << ", " << master_ids[1] << ", " << master_ids[2] << std::endl;

    // All points are marked as kept (1) by default. As they are masked by
    // neighbors within the user-specified radius, their value is changed to 0.
    std::vector<int> keep(np, 1);

    outView->appendPoint(*inView, idxZ);

    // We now proceed to mask all neighbors within m_radius of the kept
    // point.
    std::vector<PointId> ids;
    ids.clear();
    ids = index.radius(idxZ, m_maskingRadius);
    log()->get(LogLevel::Debug) << ids.size() << std::endl;
    for (auto const& id : ids)
        keep[id] = 0;
    //keep[idxZ] = 1;

    for (auto const& cur_idx : master_ids)
    {
        if (keep[cur_idx] == 0)
            continue;

        // Find nearest point to idxZ that is outside the maskingRadius.
        //auto neighbors = index.neighbors(cur_idx, ids.size()+1);
        //PointId neighbor = neighbors.back();

        auto neighbor_ids = index.radius(cur_idx, m_elevationRadius);
//        log()->get(LogLevel::Debug) << neighbor_ids.size() << std::endl;
        
        // Check that neighbor is the lowest in it's neighborhood.
        double localMin = inView->getFieldAs<double>(Dimension::Id::Z, cur_idx);
        bool isLocalMin(true);
        for (auto const& ni : neighbor_ids)
        {
//            log()->get(LogLevel::Debug) << cur_idx << ", " << ni << std::endl;
            double localZ = inView->getFieldAs<double>(Dimension::Id::Z, ni);
//            log()->get(LogLevel::Debug) << localMin << ", " << localZ << std::endl;
            if ((localMin - localZ) > m_elevationThreshold)
            {
                isLocalMin = false;
                break;
            }
        }
        if (isLocalMin)
        {
            outView->appendPoint(*inView, cur_idx);
            ids.clear();
            ids = index.radius(cur_idx, m_maskingRadius);
//            log()->get(LogLevel::Debug) << ids.size() << std::endl;
            for (auto const& id : ids)
                keep[id] = 0;
            //keep[neighbor] = 1;
        }
    }

    // Simply calculate the percentage of retained points.
    double frac = (double)outView->size() / (double)inView->size();
    log()->get(LogLevel::Debug2)
        << "Retaining " << outView->size() << " of " << inView->size()
        << " points (" << 100 * frac << "%)\n";

    viewSet.insert(outView);
    return viewSet;
}

} // namespace pdal
