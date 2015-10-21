/******************************************************************************
* Copyright (c) 2015, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "CullerFilter.hpp"

#include <random>
#include <vector>

#include "PCLConversions.hpp"
#include "PCLPipeline.h"

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.culler", "Culler filter",
               "http://pdal.io/stages/filters.culler.html");

CREATE_SHARED_PLUGIN(1, 0, CullerFilter, Filter, s_info)

std::string CullerFilter::getName() const
{
    return s_info.name;
}

Options CullerFilter::getDefaultOptions()
{
    Options options;
    options.add("tolerance", 0.1, "Tolerance");
    return options;
}

/** \brief This method processes the PointView through the given pipeline. */

void CullerFilter::processOptions(const Options& options)
{
    m_tolerance = options.getValueOrDefault<double>("tolerance", 0.1);
}

PointViewSet CullerFilter::run(PointViewPtr input)
{
    PointViewPtr output = input->makeNew();
    PointViewSet viewSet;
    viewSet.insert(output);

    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process CullerFilter..." << std::endl;

    BOX3D bounds;
    input->calculateBounds(bounds);

    // convert PointView to PointNormal
    typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
    Cloud::Ptr cloud(new Cloud);
    pclsupport::PDALtoPCD(input, *cloud, bounds);

    int level = log()->getLevel();
    switch (level)
    {
        case 0:
            pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
            break;
        case 1:
            pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
            break;
        case 2:
            pcl::console::setVerbosityLevel(pcl::console::L_WARN);
            break;
        case 3:
            pcl::console::setVerbosityLevel(pcl::console::L_INFO);
            break;
        case 4:
            pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
            break;
        default:
            pcl::console::setVerbosityLevel(pcl::console::L_VERBOSE);
            break;
    }



    typedef pcl::search::KdTree<pcl::PointXYZ> XYZKdTree;
    typedef XYZKdTree::Ptr XYZKdTreePtr;

    // Spatially indexed point cloud
    XYZKdTreePtr tree (new XYZKdTree);
    tree->setInputCloud (cloud);

    std::vector<PointId> samples;
    samples.reserve(input->size());
    std::vector<int> keep(input->size(), 0);

    std::random_device rd;

    for (int i = 0; i < cloud->points.size(); ++i)
    {
        if (keep[i] == 1)
            continue;

        std::vector<int> neighbors;
        std::vector<float> sqr_distances;
        pcl::PointXYZ temp_pt = cloud->points[i];

        int num = tree->radiusSearch(temp_pt, m_tolerance, neighbors, sqr_distances);

        if (num > 0)
        {
            log()->get(LogLevel::Debug4) << num << " neighbors within " << m_tolerance << std::endl;
            uint32_t id = Utils::uniform(0, static_cast<uint32_t>(num-1), rd());
            samples.push_back(neighbors[id]);
            keep[neighbors[id]] = 1;
        }
        else
        {
            samples.push_back(i);
            keep[i] = 1;
        }
    }

    log()->get(LogLevel::Info) << "Retaining " << samples.size() << " of " << cloud->points.size() << " points" << std::endl;

    // append samples to output
    for (auto const& s : samples)
        output->appendPoint(*input, s);



    return viewSet;
}

} // namespace pdal
