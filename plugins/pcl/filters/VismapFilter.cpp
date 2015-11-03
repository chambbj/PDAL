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

#include "VismapFilter.hpp"

#include <random>
#include <vector>

#include "PCLConversions.hpp"
#include "PCLPipeline.h"

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.vismap", "Vismap filter",
               "http://pdal.io/stages/filters.vismap.html");

CREATE_SHARED_PLUGIN(1, 0, VismapFilter, Filter, s_info)

std::string VismapFilter::getName() const
{
    return s_info.name;
}

Options VismapFilter::getDefaultOptions()
{
    Options options;
    options.add("alpha", 1.0, "Observed Sampling Radius");
    options.add("beta", 10.0, "Observer Sampling Radius");
    options.add("gamma", 1.0, "Octree Occupancy Resolution");
    options.add("delta", 1.0, "Octree Fill Resolution");
    options.add("epsilon", 1.0, "Observer Radius");
    return options;
}

/** \brief This method processes the PointView through the given pipeline. */

void VismapFilter::processOptions(const Options& options)
{
    m_alpha = options.getValueOrDefault<double>("alpha", 1.0);
    m_beta = options.getValueOrDefault<double>("beta", 10.0);
    m_gamma = options.getValueOrDefault<double>("gamma", 1.0);
    m_delta = options.getValueOrDefault<double>("delta", 1.0);
    m_epsilon = options.getValueOrDefault<double>("epsilon", 1.0);
}

void VismapFilter::addDimensions(PointLayoutPtr layout)
{
    // m_numTotalIntersectsDim = layout->registerOrAssignDim("NumTotalIntersects", Dimension::Type::Unsigned64);
    m_numRaysDim = layout->registerOrAssignDim("NumRays", Dimension::Type::Unsigned64);
    // m_numFirstIntersects = layout->registerOrAssignDim("NumFirstIntersects", Dimension::Type::Unsigned64);
    // m_numTimesSeenDim = layout->registerOrAssignDim("NumTimesSeen", Dimension::Type::Unsigned64);
    // m_meanTotalIntersectsDim = layout->registerOrAssignDim("MeanTotalIntersects", Dimension::Type::Double);
    m_meanFirstIntersectsDim = layout->registerOrAssignDim("MeanFirstIntersects", Dimension::Type::Double);
    // m_meanTimesSeenDim = layout->registerOrAssignDim("MeanTimesSeen", Dimension::Type::Double);
}

void VismapFilter::filter(PointView& input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process VismapFilter..." << std::endl;

    BOX3D bounds;
    input.calculateBounds(bounds);

    // convert PointView to PointNormal
    typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
    typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> Octree;

    Cloud::Ptr cloud(new Cloud);
    pclsupport::PDALtoPCD(std::make_shared<PointView>(input), *cloud, bounds);

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

    auto resample = [](Cloud::Ptr original, Cloud::Ptr cloud_s, Octree::Ptr tree_a, std::vector<int>* samples, double resolution){
        std::random_device rd;

        // Spatially indexed observed point cloud
        // pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree_a(resolution / std::sqrt(3));
        tree_a->setInputCloud (cloud_s);

        samples->reserve(original->size());

        samples->push_back(0);
        tree_a->addPointToCloud(original->points[0], cloud_s);

        for (int i = 1; i < original->size(); ++i)
        {
            std::vector<int> neighbors;
            std::vector<float> sqr_distances;
            pcl::PointXYZ temp_pt = original->points[i];

            int num = tree_a->radiusSearch(temp_pt, resolution, neighbors, sqr_distances, 1);

            if (num == 0)
            {
                samples->push_back(i);
                tree_a->addPointToCloud(temp_pt, cloud_s);
            }
        }
    };

    Octree::Ptr tree_a(new Octree(m_alpha / std::sqrt(3)));
    Cloud::Ptr cloud_a(new Cloud);
    std::vector<int> samples_a;
    resample(cloud, cloud_a, tree_a, &samples_a, m_alpha);
    log()->get(LogLevel::Info) << "Retaining " << samples_a.size() << " of " << cloud->size() << " points (" <<  100*(double)samples_a.size()/(double)cloud->size() << "%)" << std::endl;

    // resample
    Octree::Ptr tree_b(new Octree(m_beta / std::sqrt(3)));
    Cloud::Ptr cloud_p(new Cloud);
    std::vector<int> samples_b;
    resample(cloud, cloud_p, tree_b, &samples_b, m_beta);
    log()->get(LogLevel::Info) << "Retaining " << samples_b.size() << " of " << cloud->size() << " points (" <<  100*(double)samples_b.size()/(double)cloud->size() << "%)" << std::endl;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree2(m_gamma);
    tree2.setInputCloud (cloud);
    tree2.defineBoundingBox();
    tree2.addPointsFromInputCloud();

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree3(m_delta);
    tree3.setInputCloud (cloud);
    tree3.defineBoundingBox();
    tree3.addPointsFromInputCloud();

    // std::vector<uint64_t> numTotalIntersects(cloud->size(), 0);
    std::vector<uint64_t> numRays(cloud->size(), 0);
    std::vector<uint64_t> numFirstIntersects(cloud->size(), 0);
    // std::vector<uint64_t> numTimesSeen(cloud->size(), 0);
    // uint64_t nRays = cloud_b->size();

    for (int i = 0; i < cloud_a->size(); ++i)
    {
        // get observed point
        pcl::PointXYZ pp = cloud_a->points[i];
        Eigen::Vector3f p = pp.getVector3fMap();
        // move to observed height
        p[2] += 1.7;

        // uint64_t nTotalIntersects = 0;
        uint64_t nFirstIntersects = 0;
        uint64_t nRays = 0;

        // get points in radius
        pcl::PointIndices::Ptr neighbors (new pcl::PointIndices ());
        std::vector<float> sqr_distances;
        tree_b->radiusSearch(pp, m_epsilon, neighbors->indices, sqr_distances);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_p);
        extract.setIndices(neighbors);
        Cloud::Ptr cloud_b(new Cloud);
        extract.filter(*cloud_b);
        // log()->get(LogLevel::Debug) << neighbors->indices.size() << ", " << cloud_p->size() << std::endl;

        for (int j = 0; j < cloud_b->size(); ++j)
        {
            // I can see myself
            if (samples_a[i] == samples_b[j])
                continue;

            // this will generally be cloud_b->size()
            nRays++;

            // get observer point
            Eigen::Vector3f q = cloud_b->points[j].getVector3fMap();
            // move to observer height
            q[2] += 1.7;
            // compute direction
            Eigen::Vector3f dir(p-q);

            int num;
            pcl::PointCloud<pcl::PointXYZ>::VectorType voxelsInRay;

            // check for at most 1 intersected voxel (an occlusion)
            num = tree2.getIntersectedVoxelCenters(p, dir, voxelsInRay, 1);

            // if un-occluded, we move on to the next observer
            if (num==0)
                continue;
            // occluded, increment number of first intersects
            nFirstIntersects++;

            // // check for total number of intersected voxels (degree of occlusion)
            // // num = tree2.getIntersectedVoxelCenters(p, dir, voxelsInRay);
            // // increase tally of total occluding voxels by number currently occluding
            // nTotalIntersects += num;

            // std::vector<int> indicesInRay;
            // Eigen::Vector3f ppp = cloud_a->points[i].getVector3fMap();
            // // compute direction
            // // Eigen::Vector3f dir2(p-qq);
            // Eigen::Vector3f invdir(q-ppp);
            // tree2.getIntersectedVoxelIndices(q, invdir, indicesInRay, 1);
            // for (auto const& id : indicesInRay)
            // {
            //     numTimesSeen[id]++;
            //     // uint64_t nts = input.getFieldAs<uint64_t>(m_numTimesSeenDim, id);
            //     // input.setField(m_numTimesSeenDim, id, nts+1);
            // }
        }

        // get indices of points in the current voxel (full point cloud)
        std::vector<int> pointIdxVec;
        tree3.voxelSearch(pp, pointIdxVec);
        // record visibility metrics
        for (auto const& id : pointIdxVec)
        {
            // numTotalIntersects[id] = nTotalIntersects;
            numRays[id] = nRays;
            numFirstIntersects[id] = nFirstIntersects;
        }
    }

    // add/write the updated dimensions
    for (int i = 0; i < cloud->size(); ++i)
    {
        // input.setField(m_numTotalIntersectsDim, i, numTotalIntersects[i]);
        input.setField(m_numRaysDim, i, numRays[i]);
        // input.setField(m_numFirstIntersects, i, numFirstIntersects[i]);
        if (numRays[i] > 0)
        {
            input.setField(m_meanFirstIntersectsDim, i, (double)numFirstIntersects[i]/numRays[i]);
            // input.setField(m_meanTotalIntersectsDim, i, (double)numTotalIntersects[i]/numRays[i]);
            // input.setField(m_meanTimesSeenDim, i, (double)numTimesSeen[i]/numRays[i]);
        }
    }
}

} // namespace pdal
