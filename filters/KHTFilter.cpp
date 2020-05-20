/******************************************************************************
 * Copyright (c) 2020, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "KHTFilter.hpp"

#include "private/3dkht/ClusterNode.hpp"

#include <Eigen/Dense>

#include <numeric>

namespace pdal
{

using namespace Dimension;
using namespace Eigen;

static PluginInfo const s_info{"filters.kht", "3D-KHT filter",
                               "http://pdal.io/stages/filters.kht.html"};

CREATE_STATIC_STAGE(KHTFilter, s_info)

KHTFilter::KHTFilter() {}

KHTFilter::~KHTFilter() {}

std::string KHTFilter::getName() const
{
    return s_info.name;
}

void KHTFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::Classification);
}

void KHTFilter::cluster(PointViewPtr view, PointIdList ids, int level,
                        std::deque<ClusterNode>& nodes)
{
    // If there are too few points in the current node, then bail. We need a
    // minimum number of points to determine whether or not the points are
    // coplanar and if this cluster can be used for voting.
    if (ids.size() < 30)
        return;

    ClusterNode n(view, ids);

    // Don't even begin looking for clusters of coplanar points until we reach a
    // given level in the octree.
    if (level > 3)
    {
        n.initialize();
        Vector3d eigVal = n.eigenvalues();

        // Test plane thickness and isotropy.
        if ((eigVal[1] > 25 * eigVal[0]) && (6 * eigVal[1] > eigVal[2]))
        {
            n.m_coplanar = true;
            n.refineFit();
            nodes.push_back(n);
            log()->get(LogLevel::Debug)
                << "Approx. coplanar node #" << nodes.size() << ": "
                << ids.size() << " points at level " << level << std::endl;
            return;
        }
    }

    // Loop over children and recursively cluster at the next level in the
    // octree.
    for (PointIdList const& child : n.children())
        cluster(view, child, level + 1, nodes);

    return;
}

PointViewSet KHTFilter::run(PointViewPtr view)
{
    // Quick check that we have any points to process
    PointViewSet viewSet;
    if (!view->size())
        return viewSet;

    // Gather initial set of PointIds for the PointView
    PointIdList ids(view->size());
    std::iota(ids.begin(), ids.end(), 0);

    // Begin hierarchical clustering at level 0 using all PointIds
    std::deque<ClusterNode> nodes;
    cluster(view, ids, 0, nodes);

    m_totalArea = 0.0;
    m_totalPoints = 0;

    for (ClusterNode n : nodes)
    {
        m_totalArea += n.area();
        m_totalPoints += n.size();
    }

    log()->get(LogLevel::Debug)
        << m_totalArea << ", " << m_totalPoints << std::endl;

    for (ClusterNode n : nodes)
    {
        n.compute();
        n.vote(m_totalArea, m_totalPoints);
    }

    // Insert the clustered view and return
    viewSet.insert(view);
    return viewSet;
}

} // namespace pdal
