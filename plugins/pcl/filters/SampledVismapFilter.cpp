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

#include "SampledVismapFilter.hpp"

#include <random>
#include <vector>

#include "dart_sample.h"
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
    PluginInfo("filters.sampledvismap", "SampledVismap filter",
               "http://pdal.io/stages/filters.sampledvismap.html");

CREATE_SHARED_PLUGIN(1, 0, SampledVismapFilter, Filter, s_info)

std::string SampledVismapFilter::getName() const
{
    return s_info.name;
}

Options SampledVismapFilter::getDefaultOptions()
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

void SampledVismapFilter::processOptions(const Options& options)
{
    m_alpha = options.getValueOrDefault<double>("alpha", 1.0);
    m_beta = options.getValueOrDefault<double>("beta", 10.0);
    m_gamma = options.getValueOrDefault<double>("gamma", 1.0);
    m_delta = options.getValueOrDefault<double>("delta", 1.0);
    m_epsilon = options.getValueOrDefault<double>("epsilon", 100.0);
}

void SampledVismapFilter::addDimensions(PointLayoutPtr layout)
{
    m_numRaysDim = layout->registerOrAssignDim("NumRays", Dimension::Type::Unsigned64);
    m_meanOcclusionsDim = layout->registerOrAssignDim("MeanOcclusions", Dimension::Type::Double);
}

void SampledVismapFilter::filter(PointView& input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process SampledVismapFilter..." << std::endl;

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

    // pick observed points from the full point cloud, with minimum distance m_alpha
    std::vector<int> observed_samples;
    observed_samples.reserve(cloud->size());
    pcl::DartSample<pcl::PointXYZ> ds;
    ds.setInputCloud(cloud);
    ds.setRadius(m_alpha);
    ds.filter(observed_samples);
    log()->get(LogLevel::Info) << "Retaining " << observed_samples.size()
        << " observed points of " << cloud->size() << " points ("
        <<  100 * (double)observed_samples.size() / (double)cloud->size()
        << "%)" << std::endl;

    // create tree for down-selecting observers, based off full cloud
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree(m_epsilon);
    tree.setInputCloud (cloud);
    tree.defineBoundingBox();
    tree.addPointsFromInputCloud();

    // create tree for testing for intersected voxels, based off full cloud
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree2(m_gamma);
    tree2.setInputCloud (cloud);
    tree2.defineBoundingBox();
    tree2.addPointsFromInputCloud();

    // create tree for painting visibility metrics, based off full cloud
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree3(m_delta);
    tree3.setInputCloud (cloud);
    tree3.defineBoundingBox();
    tree3.addPointsFromInputCloud();

    // for each observed location, count number of rays and number of occlusions
    std::vector<uint64_t> numRays(cloud->size(), 0);
    std::vector<uint64_t> numOcclusions(cloud->size(), 0);
    for (auto const& observed_idx : observed_samples)
    {
        // get observed point
        pcl::PointXYZ observed_pt = cloud->points[observed_idx];
        Eigen::Vector3f p = observed_pt.getVector3fMap();
        // move to observed height
        p[2] += 1.7;

        uint64_t nOcclusions = 0;
        uint64_t nRays = 0;

        // get points within radius of observed point from full cloud
        pcl::PointIndices::Ptr neighbors (new pcl::PointIndices ());
        std::vector<float> sqr_distances;
        tree.radiusSearch(observed_pt, m_epsilon, neighbors->indices, sqr_distances);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(neighbors);
        Cloud::Ptr cloud_c(new Cloud);
        extract.filter(*cloud_c);

        // pick observer points from the points in radius, with minimum distance m_beta
        std::vector<int> observer_samples;
        observer_samples.reserve(cloud_c->size());
        ds.setInputCloud(cloud_c);
        ds.setRadius(m_beta);
        ds.filter(observer_samples);
        // log()->get(LogLevel::Info) << "Retaining " << observer_samples.size()
        //     << " observer points of " << cloud_c->size() << " points ("
        //     <<  100 * (double)observer_samples.size() / (double)cloud_c->size()
        //     << "%)" << std::endl;

        for (auto const& observer_idx : observer_samples)
        {
            // I can see myself
            if (observed_idx == neighbors->indices[observer_idx])
                continue;

            // this will generally be cloud_c->size()
            nRays++;

            // get observer point
            Eigen::Vector3f q = cloud->points[neighbors->indices[observer_idx]].getVector3fMap();
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
            nOcclusions++;
        }

        // get indices of points in the current voxel (full point cloud)
        std::vector<int> pointIdxVec;
        tree3.voxelSearch(observed_pt, pointIdxVec);
        // record visibility metrics
        for (auto const& id : pointIdxVec)
        {
            numRays[id] = nRays;
            numOcclusions[id] = nOcclusions;
        }
    }

    // add/write the updated dimensions
    for (int i = 0; i < cloud->size(); ++i)
    {
        input.setField(m_numRaysDim, i, numRays[i]);
        if (numRays[i] > 0)
        {
            input.setField(m_meanOcclusionsDim, i,
                (double)numOcclusions[i] / numRays[i]);
        }
    }
}

} // namespace pdal
