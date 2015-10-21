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

#include "ProgressiveDartFilter.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "PCLConversions.hpp"
#include "PCLPipeline.h"

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/octree/octree.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.progressivedart", "ProgressiveDart filter",
               "http://pdal.io/stages/filters.progressivedart.html");

CREATE_SHARED_PLUGIN(1, 0, ProgressiveDartFilter, Filter, s_info)

std::string ProgressiveDartFilter::getName() const
{
    return s_info.name;
}

void ProgressiveDartFilter::addDimensions(PointLayoutPtr layout)
{
    m_lodDim = layout->registerOrAssignDim("LOD", Dimension::Type::Unsigned8);
    // layout->registerDim(Dimension::Id::Classification);
}

Options ProgressiveDartFilter::getDefaultOptions()
{
    Options options;
    options.add("tolerance", 0.1, "Tolerance");
    return options;
}

/** \brief This method processes the PointView through the given pipeline. */

void ProgressiveDartFilter::processOptions(const Options& options)
{
    m_tolerance = options.getValueOrDefault<double>("tolerance", 0.1);
}

PointViewSet ProgressiveDartFilter::run(PointViewPtr input)
{
    PointViewPtr output = input->makeNew();
    PointViewSet viewSet;
    viewSet.insert(output);

    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process ProgressiveDartFilter..." << std::endl;

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


    pcl::search::KdTree<pcl::PointXYZ> tree1;
    tree1.setInputCloud (cloud);
    double dists = 0.0;
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        std::vector<int> neighbors;
        std::vector<float> sqr_distances;
        pcl::PointXYZ temp_pt = cloud->points[i];

        int num = tree1.nearestKSearch(temp_pt, 2, neighbors, sqr_distances);
        dists += std::sqrt(sqr_distances[1]);
        // log()->get(LogLevel::Debug2) << i << ", " << neighbors[1] << ", "
        //                             << num << ", " << sqr_distances[1] << std::endl;
    }
    // log()->get(LogLevel::Debug) << dists << std::endl;
    log()->get(LogLevel::Debug) << dists/cloud->points.size() << std::endl;


    // Spatially indexed point cloud
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree(m_tolerance / std::sqrt(3));
    Cloud::Ptr cloud_t(new Cloud);
    tree.setInputCloud (cloud_t);

    std::vector<int> samples;
    samples.reserve(input->size());

    std::random_device rd;

    samples.push_back(0);
    tree.addPointToCloud(cloud->points[0], cloud_t);

    std::vector<int> remaining;
    remaining.reserve(input->size()-1);
    for (int j = 0; j < input->size()-1; ++j)
        remaining.push_back(j+1);
        // remaining[j] = j+1;
    std::vector<int> temp;
    temp.reserve(remaining.size());

    int beg = 0;
    int end;

    int dim = 2;
    // double A = std::log(std::sqrt(cloud->points.size())-1)/std::log(2);
    double A = std::log(std::pow(cloud->points.size(), 1.0/dim)-1)/std::log(2);
    // log()->get(LogLevel::Debug) << A << std::endl;
    double B = 1 / (std::pow(2.0, A));
    // log()->get(LogLevel::Debug) << B << std::endl;
    double C = m_tolerance / B * std::sqrt(dim);
    // log()->get(LogLevel::Debug) << C << std::endl;

    int num_levels = 20;
    std::vector<point_count_t> pts_arr(num_levels, 0);
    std::vector<double> lev_arr(num_levels, 0.0);
    std::vector<double> res_arr(num_levels, 0.0);
    std::vector<double> rad_arr(num_levels, 0.0);
    pts_arr[0] = cloud->points.size();
    for (int j = 1; j < num_levels; ++j)
    {
        pts_arr[j] = std::ceil(pts_arr[0] * (1.0 - static_cast<double>(j)/static_cast<double>(num_levels)));
        // lev_arr[j] = std::log(std::sqrt(pts_arr[j])-1)/std::log(2);
        lev_arr[j] = std::log(std::pow(pts_arr[j], 1.0/dim)-1)/std::log(2);
        res_arr[j] = 1 / (std::pow(2.0, lev_arr[j]));
        // rad_arr[j] = res_arr[j] * C / std::sqrt(dim);
        rad_arr[j] = m_tolerance * (static_cast<double>(j)/num_levels);
    }

    for (int j = num_levels-1; j > 0; --j)
    {
        log()->get(LogLevel::Debug) << pts_arr[j] << ", " << lev_arr[j] << ", " << res_arr[j] << ", " << rad_arr[j] << std::endl;
        temp.clear();
        for (int i = 0; i < remaining.size(); ++i)
        {
            std::vector<int> neighbors;
            std::vector<float> sqr_distances;
            pcl::PointXYZ temp_pt = cloud->points[remaining[i]];

            int num = tree.radiusSearch(temp_pt, rad_arr[j], neighbors, sqr_distances);

            if (num == 0)
            {
                samples.push_back(remaining[i]);
                tree.addPointToCloud(temp_pt, cloud_t);
            }
            else
            {
                temp.push_back(remaining[i]);
            }
        }
        remaining.swap(temp);

        log()->get(LogLevel::Info) << "Retaining " << samples.size() << " of " << cloud->points.size() << " points" << std::endl;
        log()->get(LogLevel::Info) << "   (adding " << samples.size() - beg << " samples, " << remaining.size() << " remaining)" << std::endl;

        // // append samples to output
        // for (auto const& s : samples)
        //     output->appendPoint(*input, s);

        end = samples.size();

        for (int i = beg; i < end; ++i)
        {
            input->setField(m_lodDim, samples[i], j);
            // input->setField(Dimension::Id::Classification, samples[i], j);
            output->appendPoint(*input, samples[i]);

            // PointId idx = output->size()-1;
            // uint8_t u = output->getFieldAs<uint8_t>(m_lodDim, idx);
            // // std::cerr << j << " : " << u << std::endl;
            // assert(j == u);
        }

        beg = end;

    }

    // need one more step to add remaining
    for (int i = 0; i < remaining.size(); ++i)
    {
      input->setField(m_lodDim, remaining[i], 0);
      // input->setField(Dimension::Id::Classification, samples[i], j);
      output->appendPoint(*input, remaining[i]);
    }

    return viewSet;
}

} // namespace pdal
