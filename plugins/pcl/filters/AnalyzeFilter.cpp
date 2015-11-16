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

#include "AnalyzeFilter.hpp"

#include "PCLConversions.hpp"
#include <pdal/StageFactory.hpp>

// other
// #include <pcl/console/print.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace pdal
{

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;

static PluginInfo const s_info = PluginInfo(
    "filters.analyze",
    "Analyze spatial distribution of points.",
    "http://pdal.io/stages/filters.analyze.html" );

CREATE_SHARED_PLUGIN(1, 0, AnalyzeFilter, Filter, s_info)

std::string AnalyzeFilter::getName() const
{
    return s_info.name;
}

void AnalyzeFilter::addDimensions(PointLayoutPtr layout)
{
    // m_density3dDim = layout->registerOrAssignDim("Density3d", Dimension::Type::Double);
    m_spacing3dDim = layout->registerOrAssignDim("Spacing3d", Dimension::Type::Double);
    // m_density2dDim = layout->registerOrAssignDim("Density2d", Dimension::Type::Double);
    m_spacing2dDim = layout->registerOrAssignDim("Spacing2d", Dimension::Type::Double);
}

void AnalyzeFilter::filter(PointView& view)
{
  BOX3D bounds;
  view.calculateBounds(bounds);

  Cloud::Ptr cloud_in(new Cloud);
  pclsupport::PDALtoPCD(std::make_shared<PointView>(view), *cloud_in, bounds);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
  tree.reset(new pcl::search::KdTree<pcl::PointXYZ> (false));
  tree->setInputCloud(cloud_in);

  double radius = 1.0;
  double area3d = 4 * 3.14159 * radius * radius;
  double area2d = 3.14159 * radius * radius;
  double density2dSum = 0.0;
  double spacing2dSum = 0.0;
  double density3dSum = 0.0;
  double spacing3dSum = 0.0;

  // Create a set of planar coefficients with X=Y=0,Z=1
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  coefficients->values.resize(4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);

  Cloud::Ptr cloud_p(new Cloud);
  proj.setInputCloud(cloud_in);
  proj.setModelCoefficients(coefficients);
  proj.filter(*cloud_p);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_p;
  tree_p.reset(new pcl::search::KdTree<pcl::PointXYZ> (false));
  tree_p->setInputCloud(cloud_p);

  for (int i = 0; i < cloud_in->size(); ++i)
  {
      pcl::PointXYZ query = cloud_in->points[i];

      std::vector<int> neighbors;
      std::vector<float> sqr_distances;
      // int num = tree->radiusSearch(query, radius, neighbors, sqr_distances);
      // // num++;
      // // log()->get(LogLevel::Debug) << "3d: " << sqr_distances[0] << std::endl;
      // view.setField(m_density3dDim, i, static_cast<double>(num)/area3d);
      // density3dSum += static_cast<double>(num)/area3d;

      neighbors.resize(2, 0);
      sqr_distances.resize(2, 0.0);
      tree->nearestKSearch(query, 2, neighbors, sqr_distances);
      view.setField(m_spacing3dDim, i, std::sqrt(sqr_distances[1]));
      // view.setField(m_density3dDim, i, 1.0 / sqr_distances[1]);
      spacing3dSum += std::sqrt(sqr_distances[1]);

      query = cloud_p->points[i];

      // neighbors.clear();
      // sqr_distances.clear();
      // num = tree_p->radiusSearch(query, radius, neighbors, sqr_distances);
      // // num++;
      // // log()->get(LogLevel::Debug) << "2d: " << sqr_distances[0] << std::endl;
      // view.setField(m_density2dDim, i, static_cast<double>(num)/area2d);
      // density2dSum += static_cast<double>(num)/area2d;

      neighbors.resize(2, 0);
      sqr_distances.resize(2, 0.0);
      tree_p->nearestKSearch(query, 2, neighbors, sqr_distances);
      view.setField(m_spacing2dDim, i, std::sqrt(sqr_distances[1]));
      // view.setField(m_density2dDim, i, 1.0 / (sqr_distances[1] + 0.000001));
      spacing2dSum += std::sqrt(sqr_distances[1]);
  }

  double density2d = 1.0;
  density2d /= spacing2dSum / cloud_in->size();
  density2d *= density2d;

  double density3d = 1.0;
  density3d /= spacing3dSum / cloud_in->size();
  density3d *= density3d;

  // log()->get(LogLevel::Debug) << density3dSum / cloud_in->size() << std::endl;
  log()->get(LogLevel::Debug) << spacing3dSum / cloud_in->size() << ", " << density3d << std::endl;
  // log()->get(LogLevel::Debug) << density2dSum / cloud_in->size() << std::endl;
  log()->get(LogLevel::Debug) << spacing2dSum / cloud_in->size() << ", " << density2d << std::endl;
}

} // namespace pdal
