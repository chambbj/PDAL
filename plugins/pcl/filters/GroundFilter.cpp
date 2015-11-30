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

#include "GroundFilter.hpp"

#include <future>
#include <random>

#include "dart_sample.h"
#include "PCLConversions.hpp"

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>

#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/approximate_progressive_morphological_filter.h>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.ground", "Progressive morphological filter",
               "http://pdal.io/stages/filters.ground.html");

CREATE_SHARED_PLUGIN(1, 0, GroundFilter, Filter, s_info)

std::string GroundFilter::getName() const
{
    return s_info.name;
}

Options GroundFilter::getDefaultOptions()
{
    Options options;
    options.add("max_window_size", 33, "Maximum window size");
    options.add("slope", 1, "Slope");
    options.add("max_distance", 2.5, "Maximum distance");
    options.add("initial_distance", 0.15, "Initial distance");
    options.add("cell_size", 1, "Cell Size");
    options.add("classify", true, "Apply classification labels?");
    options.add("extract", false, "Extract ground returns?");
    options.add("approximate", false, "Use approximate algorithm?");
    options.add("num_iters", 10, "Number of iterations");
    options.add("mean_res", 1.0, "Mean resolution");
    return options;
}

void GroundFilter::processOptions(const Options& options)
{
    m_maxWindowSize = options.getValueOrDefault<double>("max_window_size", 33);
    m_slope = options.getValueOrDefault<double>("slope", 1);
    m_maxDistance = options.getValueOrDefault<double>("max_distance", 2.5);
    m_initialDistance = options.getValueOrDefault<double>("initial_distance", 0.15);
    m_cellSize = options.getValueOrDefault<double>("cell_size", 1);
    m_classify = options.getValueOrDefault<bool>("classify", true);
    m_extract = options.getValueOrDefault<bool>("extract", false);
    m_approximate = options.getValueOrDefault<bool>("approximate", false);
    m_numIters = options.getValueOrDefault<int>("num_iters", 10);
    m_meanRes = options.getValueOrDefault<double>("mean_res", 1.0);
}

void GroundFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
    m_ppmfDim = layout->registerOrAssignDim("PPMF", Dimension::Type::Double);
}

PointViewSet GroundFilter::run(PointViewPtr input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process GroundFilter...\n";

    // convert PointView to PointXYZ
    typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
    Cloud::Ptr cloud(new Cloud);
    BOX3D bounds;
    input->calculateBounds(bounds);
    pclsupport::PDALtoPCD(input, *cloud, bounds);

    // PCL should provide console output at similar verbosity level as PDAL
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

    PointViewPtr output = input->makeNew();
    PointViewSet viewSet;
    viewSet.insert(output);

    std::vector<int> counts(cloud->size(), 0);

    std::vector<std::future<std::vector<int>>> futures;

    for (int i = 0; i < m_numIters; ++i)
    {
        auto is_ground = [](double m_meanRes, Cloud::Ptr cloud,
            bool m_approximate, double m_maxWindowSize, double m_slope,
            double m_maxDistance, double m_initialDistance, double m_cellSize)
        {

          auto clamp = [](double t, double min, double max)
	        {
	            return ((t < min) ? min : ((t > max) ? max : t));
	        };

          std::random_device rd;
          std::mt19937 gen(rd());

          std::uniform_real_distribution<> dis(0.3, m_meanRes);
          double rad = dis(gen);

          std::normal_distribution<double> dis1(m_maxWindowSize, 15.0);
          double winSize = clamp(dis1(gen), 1.0, 100.0);

          std::normal_distribution<double> dis2(m_slope, 15.0);
          double slope = std::tan(clamp(dis2(gen), 0.0, 90.0)*3.14159/180.0);

          std::normal_distribution<double> dis3(m_maxDistance, 4.0);
          double maxDist = clamp(dis3(gen), 0.01, 100.0);

          std::normal_distribution<double> dis4(m_initialDistance, 1.0);
          double initDist = clamp(dis4(gen), 0.01, 1.0);

          std::normal_distribution<double> dis5(m_cellSize, 1.0);
          double cellSize = clamp(dis5(gen), 0.5, 2.0);

          std::cerr << "Process with " << rad
                    << ", " << winSize
                    << ", " << slope
                    << ", " << maxDist
                    << ", " << initDist
                    << ", " << cellSize << std::endl;

          pcl::DartSample<pcl::PointXYZ> ds;
          ds.setInputCloud(cloud);
          ds.setRadius(rad);

          // std::vector<int> samples;
          pcl::PointIndices::Ptr samples(new pcl::PointIndices());
          ds.filter(samples->indices);

          pcl::ExtractIndices<pcl::PointXYZ> extract;
          extract.setInputCloud(cloud);
          extract.setIndices(samples);

          Cloud::Ptr cloud_sampled(new Cloud);
          extract.setNegative(false);
          extract.filter(*cloud_sampled);

          // setup the PMF filter
          pcl::PointIndicesPtr idx(new pcl::PointIndices);
          if (!m_approximate)
          {

              pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
              pmf.setInputCloud(cloud_sampled);
              pmf.setMaxWindowSize(winSize);
              pmf.setSlope(slope);
              pmf.setMaxDistance(maxDist);
              pmf.setInitialDistance(initDist);
              pmf.setCellSize(cellSize);

              // run the PMF filter, grabbing indices of ground returns
              pmf.extract(idx->indices);
          }
          else
          {
              pcl::ApproximateProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
              pmf.setInputCloud(cloud_sampled);
              pmf.setMaxWindowSize(winSize);
              pmf.setSlope(slope);
              pmf.setMaxDistance(maxDist);
              pmf.setInitialDistance(initDist);
              pmf.setCellSize(cellSize);

              // run the PMF filter, grabbing indices of ground returns
              pmf.extract(idx->indices);
          }

          std::vector<int> retval(idx->indices.size(), 0);
          int j = 0;
          for (auto const& i : idx->indices)
              retval[j++] = samples->indices[i];

          return retval;
        };

        futures.push_back(std::async(is_ground, m_meanRes, cloud,
            m_approximate, m_maxWindowSize, m_slope,
            m_maxDistance, m_initialDistance, m_cellSize));
      }

      for (auto& fut : futures) {
        std::vector<int> samples = fut.get();
        for (auto const& i : samples)
          counts[i]++;
      }

      for (PointId i = 0; i < cloud->size(); ++i)
      {
          input->setField(m_ppmfDim, i, (double)counts[i]/m_numIters);
          output->appendPoint(*input, i);
      }

    return viewSet;
}

} // namespace pdal
