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

#include "ImportanceDartFilter.hpp"

#include <random>
#include <vector>

#include "PCLConversions.hpp"
#include "PCLPipeline.h"

#include <pcl/console/print.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.importancedart", "ImportanceDart filter",
               "http://pdal.io/stages/filters.importancedart.html");

CREATE_SHARED_PLUGIN(1, 0, ImportanceDartFilter, Filter, s_info)

std::string ImportanceDartFilter::getName() const
{
    return s_info.name;
}

Options ImportanceDartFilter::getDefaultOptions()
{
    Options options;
    options.add("tolerance", 0.1, "Tolerance");
    options.add("radius", 1.0, "Radius");
    options.add("sigma", 1.0, "Sigma");
    return options;
}

void ImportanceDartFilter::addDimensions(PointLayoutPtr layout)
{
    m_diffMeanDim = layout->registerOrAssignDim("DiffMean", Dimension::Type::Double);
    m_toleranceDim = layout->registerOrAssignDim("Tolerance", Dimension::Type::Double);
    m_aglDim = layout->registerOrAssignDim("AGL", Dimension::Type::Double);
}

/** \brief This method processes the PointView through the given pipeline. */

void ImportanceDartFilter::processOptions(const Options& options)
{
    m_tolerance = options.getValueOrDefault<double>("tolerance", 0.1);
    m_radius = options.getValueOrDefault<double>("radius", 1.0);
    m_sigma = options.getValueOrDefault<double>("sigma", 1.0);
}

PointViewSet ImportanceDartFilter::run(PointViewPtr input)
{
    PointViewPtr output = input->makeNew();
    PointViewSet viewSet;
    viewSet.insert(output);

    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process ImportanceDartFilter..." << std::endl;

    BOX3D bounds;
    input->calculateBounds(bounds);

    int cols = (int)(ceil((bounds.maxx - bounds.minx)/m_radius)) + 1;
    int rows = (int)(ceil((bounds.maxy - bounds.miny)/m_radius)) + 1;

    Eigen::MatrixXi counts = Eigen::MatrixXi::Zero(rows, cols);
    Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd mean = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::MatrixXd diff = Eigen::MatrixXd::Zero(rows, cols);

    log()->get(LogLevel::Debug) << "Binning points..." << std::endl;

    for (PointId id = 0; id < input->size(); ++id)
    {
        using namespace Dimension::Id;
        double x = input->getFieldAs<double>(X, id);
        double y = input->getFieldAs<double>(Y, id);
        double z = input->getFieldAs<double>(Z, id);

        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int col = clamp(static_cast<int>(floor((x - bounds.minx) / m_radius)), 0, cols-1);
        int row = clamp(static_cast<int>(floor((bounds.maxy - y) / m_radius)), 0, rows-1);

        counts(row, col)++;
        sum(row, col) += z;
    }

    log()->get(LogLevel::Debug) << "Computing mean..." << std::endl;

    for (int c = 0; c < cols; ++c)
    {
        for (int r = 0; r < rows; ++r)
        {
            mean(r, c) = sum(r, c) / counts(r, c);
        }
    }

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



    // Spatially indexed point cloud
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> tree(m_tolerance / std::sqrt(3));
    Cloud::Ptr cloud_t(new Cloud);
    tree.setInputCloud (cloud_t);

    std::vector<int> samples;
    samples.reserve(input->size());

    std::random_device rd;

    samples.push_back(0);
    tree.addPointToCloud(cloud->points[0], cloud_t);

    log()->get(LogLevel::Debug) << "Sampling points..." << std::endl;

    for (int i = 1; i < cloud->points.size(); ++i)
    {
        using namespace Dimension::Id;
        double x = input->getFieldAs<double>(X, i);
        double y = input->getFieldAs<double>(Y, i);
        double z = input->getFieldAs<double>(Z, i);
        double a = input->getFieldAs<double>(m_aglDim, i);

        auto clamp = [](double t, double min, double max)
        {
            return ((t < min) ? min : ((t > max) ? max : t));
        };

        int col = clamp(static_cast<int>(floor((x - bounds.minx) / m_radius)), 0, cols-1);
        int row = clamp(static_cast<int>(floor((bounds.maxy - y) / m_radius)), 0, rows-1);

        // // try adaptive radius vice m_tolerance
        // double arad = m_tolerance;
        // double diff = std::fabs(z - mean(row, col));
        // // if (diff > 0.0)
        // //     arad *= std::exp(-(diff*diff));
        // arad *= std::exp(-(diff*diff)/(2*m_sigma*m_sigma));
        // log()->get(LogLevel::Debug4) << "Mean: " << mean(row, col)
        //                              << ", diff: " << diff
        //                              << ", radius: " << arad << std::endl;
        //
        // input->setField(m_diffMeanDim, i, z - mean(row, col));
        // input->setField(m_toleranceDim, i, arad);

        double arad = m_tolerance;
        if (a > 1.0)
            arad *= 0.5;

        std::vector<int> neighbors;
        std::vector<float> sqr_distances;
        pcl::PointXYZ temp_pt = cloud->points[i];

        int num = tree.radiusSearch(temp_pt, arad, neighbors, sqr_distances);

        if (num == 0)
        {
            samples.push_back(i);
            tree.addPointToCloud(temp_pt, cloud_t);
        }
    }

    log()->get(LogLevel::Info) << "Retaining " << samples.size() << " of " << cloud->points.size() << " points (" <<  100*(double)samples.size()/(double)cloud->points.size() << "%)" << std::endl;

    // append samples to output
    for (auto const& s : samples)
        output->appendPoint(*input, s);



    return viewSet;
}

} // namespace pdal
