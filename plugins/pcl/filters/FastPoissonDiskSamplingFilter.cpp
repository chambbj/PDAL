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

#include "FastPoissonDiskSamplingFilter.hpp"

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
    PluginInfo("filters.fastpoissondisksampling", "Fast Poisson Disk Sampling filter",
               "http://pdal.io/stages/filters.fastpoissondisksampling.html");

CREATE_SHARED_PLUGIN(1, 0, FastPoissonDiskSamplingFilter, Filter, s_info)

std::string FastPoissonDiskSamplingFilter::getName() const
{
    return s_info.name;
}

Options FastPoissonDiskSamplingFilter::getDefaultOptions()
{
    Options options;
    options.add("radius", 1.0, "Radius");
    options.add("neighbors", 30, "Neighbors");
    options.add("samples", 25, "Samples");
    return options;
}

/** \brief This method processes the PointView through the given pipeline. */

void FastPoissonDiskSamplingFilter::processOptions(const Options& options)
{
    m_r = options.getValueOrDefault<double>("radius", 1.0);
    m_k = options.getValueOrDefault<point_count_t>("neighbors", 30);
    m_num_samples = options.getValueOrDefault<point_count_t>("samples", 25);
}

PointViewSet FastPoissonDiskSamplingFilter::run(PointViewPtr input)
{
    PointViewPtr output = input->makeNew();
    PointViewSet viewSet;
    viewSet.insert(output);

    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);

    log()->get(LogLevel::Debug2) << "Process FastPoissonDiskSamplingFilter..." << std::endl;

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



    // http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    std::vector<PointId> samples;
    samples.reserve(input->size());
    std::vector<int> active_list;
    active_list.reserve(input->size());
    int num_dims = 3; // XYZ
    double cell_size = m_r / std::sqrt(num_dims);
    log()->get(LogLevel::Debug) << "Computed cell size is " << cell_size << std::endl;

    double xrange = bounds.maxx - bounds.minx;
    double yrange = bounds.maxy - bounds.miny;
    double zrange = bounds.maxz - bounds.minz;
    uint64_t xsamples = std::ceil(xrange / cell_size)+1;
    uint64_t ysamples = std::ceil(yrange / cell_size)+1;
    uint64_t zsamples = std::ceil(zrange / cell_size)+1;
    std::vector<int> background(xsamples*ysamples*zsamples, -1);
    log()->get(LogLevel::Debug) << "Computed samples are " << xsamples << "x" << ysamples << "x" << zsamples << std::endl;

    std::random_device rd;
    PointId id = Utils::uniform(0, input->size(), rd());
    samples.push_back(id);
    active_list.push_back(0);

    auto clamp = [](double t, double min, double max)
    {
        return ((t < min) ? min : ((t > max) ? max : t));
    };

    pcl::PointXYZ initial_sample = cloud->points[id];
    int xi = clamp(static_cast<int>(floor((initial_sample.x) / cell_size)), 0, xsamples-1);
    int yi = clamp(static_cast<int>(floor((initial_sample.y) / cell_size)), 0, ysamples-1);
    int zi = clamp(static_cast<int>(floor((initial_sample.z) / cell_size)), 0, zsamples-1);
    int k = xi + ysamples * (yi + zsamples * zi);
    background[k] = 0;
    log()->get(LogLevel::Debug) << "Randomly initialized with id " << id << std::endl;
    log()->get(LogLevel::Debug) << initial_sample << std::endl;
    // log()->get(LogLevel::Debug) << bounds << std::endl;
    // log()->get(LogLevel::Debug) << xi << ", " << yi << ", " << zi << std::endl;
    // log()->get(LogLevel::Debug) << k << std::endl;

    // search for k points in spherical annulus [r,2r]
    typedef pcl::search::KdTree<pcl::PointXYZ> XYZKdTree;
    typedef XYZKdTree::Ptr XYZKdTreePtr;

    // Spatially indexed point cloud
    XYZKdTreePtr tree (new XYZKdTree);
    tree->setInputCloud (cloud);

    while (!active_list.empty() && (samples.size() < m_num_samples))
    {
        // choose random sample from active list
        uint32_t active_list_sample = Utils::uniform(0, static_cast<uint32_t>(active_list.size())-1, rd());

        // log()->get(LogLevel::Debug2) << "Randomly selected id " << active_list_sample << " from the active list" << std::endl;
        // log()->get(LogLevel::Debug2) << "  whose value is " << active_list[active_list_sample] << std::endl;
        // log()->get(LogLevel::Debug2) << "  and index is " << samples[active_list[active_list_sample]] << std::endl;

        // Start by finding neighbors within 2x radius
        std::vector<int> neighbors;
        std::vector<float> sqr_distances;
        pcl::PointXYZ temp_pt = cloud->points[samples[active_list[active_list_sample]]];
        int num_neighbors = tree->radiusSearch (temp_pt, 2*m_r, neighbors, sqr_distances);
        log()->get(LogLevel::Debug2) << "Found " << num_neighbors << " neighbors" << std::endl;

        for (auto const& n : neighbors)
            log()->get(LogLevel::Debug3) << n << ": " << cloud->points[n] << std::endl;

        // Then remove those within radius
        std::vector<int> good_neighbors;
        // log()->get(LogLevel::Debug2) << "Initial good_neighbors size " << good_neighbors.size() << std::endl;
        for (int i = 0; i < num_neighbors; ++i)
        {
            if (sqr_distances[i] >= (m_r*m_r))
            {
                good_neighbors.push_back(neighbors[i]);
                // log()->get(LogLevel::Debug3) << neighbors[i] << ": "
                //                              << cloud->points[neighbors[i]] << std::endl;
                // log()->get(LogLevel::Debug3) << "PDAL tells me (" << input->getFieldAs<double>(Dimension::Id::X, neighbors[i])
                //                              << "," << input->getFieldAs<double>(Dimension::Id::Y, neighbors[i])
                //                              << "," << input->getFieldAs<double>(Dimension::Id::Z, neighbors[i])
                //                              << ")" << std::endl;
            }
        }
        log()->get(LogLevel::Debug2) << "Found " << good_neighbors.size() << " within annulus" << std::endl;

        for (auto const& gn : good_neighbors)
            log()->get(LogLevel::Debug3) << gn << ": " << cloud->points[gn] << std::endl;

        //     std::cerr << gn << " ";
        // std::cerr << std::endl;

        bool found_sample = false;

        // check for a sample sufficiently far away from existing samples
        PointId x;
        int iter = 0;
        log()->get(LogLevel::Debug2) << "Checking neigbors" << std::endl;
        for (auto const& gn : good_neighbors)
        {
            x = gn;
            pcl::PointXYZ neighbor_sample = cloud->points[gn];

            std::vector<uint64_t> j(num_dims), jmin(num_dims), jmax(num_dims);
            jmin[0] = clamp(static_cast<uint64_t>(floor((neighbor_sample.x - m_r) / cell_size)), 0, xsamples-1);
            jmax[0] = clamp(static_cast<uint64_t>(floor((neighbor_sample.x + m_r) / cell_size)), 0, xsamples-1);
            jmin[1] = clamp(static_cast<uint64_t>(floor((neighbor_sample.y - m_r) / cell_size)), 0, ysamples-1);
            jmax[1] = clamp(static_cast<uint64_t>(floor((neighbor_sample.y + m_r) / cell_size)), 0, ysamples-1);
            jmin[2] = clamp(static_cast<uint64_t>(floor((neighbor_sample.z - m_r) / cell_size)), 0, zsamples-1);
            jmax[2] = clamp(static_cast<uint64_t>(floor((neighbor_sample.z + m_r) / cell_size)), 0, zsamples-1);

            if (iter > m_k)
                goto done_j_loop;

            ++iter;

            log()->get(LogLevel::Debug3) << gn << ": " << neighbor_sample << std::endl;
            // log()->get(LogLevel::Debug4) << bounds << std::endl;

            for (j=jmin;;)
            {
                // log()->get(LogLevel::Debug4) << j[0] << ", " << j[1] << ", " << j[2] << std::endl;
                int k = j[0] + ysamples * (j[1] + zsamples * j[2]);
                // log()->get(LogLevel::Debug4) << k << std::endl;
                if (background[k] >= 0 && background[k] != active_list_sample)
                {
                    log()->get(LogLevel::Debug4) << background[k] << ", " << active_list_sample << std::endl;
                    log()->get(LogLevel::Debug4) << "has a point" << std::endl;
                    // if too close, reject
                    pcl::PointXYZ existing_sample = cloud->points[samples[background[k]]];
                    double dist = ((neighbor_sample.x-existing_sample.x)*(neighbor_sample.x-existing_sample.x)+(neighbor_sample.y-existing_sample.y)*(neighbor_sample.y-existing_sample.y)+(neighbor_sample.z-existing_sample.z)*(neighbor_sample.z-existing_sample.z));
                    log()->get(LogLevel::Debug4) << dist << " : " << m_r*m_r << std::endl;
                    if (dist < (m_r*m_r))
                        goto reject_sample;
                }
                // move on to next j
                for(unsigned int i=0; i<num_dims; ++i){
                   ++j[i];
                   if(j[i]<=jmax[i]){
                      break;
                   }else{
                      if(i==num_dims-1) goto done_j_loop;
                      else j[i]=jmin[i]; // and try incrementing the next dimension along
                   }
                }
            }
            done_j_loop:
            // if we made it here, we're good!
            found_sample=true;
            break;
            // if we goto here, x is too close to an existing sample
            reject_sample:
            ; // nothing to do except go to the next iteration in this loop
        }
        if(found_sample)
        {
           size_t new_sample_index = samples.size(); // the index of the new sample
           samples.push_back(x);
           active_list.push_back(new_sample_index);

           pcl::PointXYZ new_sample  = cloud->points[x];
           int xi = clamp(static_cast<int>(floor((new_sample.x) / cell_size)), 0, xsamples-1);
           int yi = clamp(static_cast<int>(floor((new_sample.y) / cell_size)), 0, ysamples-1);
           int zi = clamp(static_cast<int>(floor((new_sample.z) / cell_size)), 0, zsamples-1);
           int k = xi + ysamples * (yi + zsamples * zi);
           background[k] = (int)new_sample_index;
           log()->get(LogLevel::Debug2) << "Adding id " << x << " at " << new_sample_index << std::endl;
           log()->get(LogLevel::Debug2) << new_sample << std::endl;
          //  log()->get(LogLevel::Debug2) << xi << ", " << yi << ", " << zi << std::endl;
          //  log()->get(LogLevel::Debug2) << k << std::endl;
        }
        else
        {
           log()->get(LogLevel::Debug2) << "Could not add sample, removing " << active_list_sample << " from the list" << std::endl;
           // since we couldn't find a sample on p's disk, we remove p from the active list
          //  active_list.erase(active_list.begin() + active_list_sample);
          active_list[active_list_sample]=active_list.back();
          active_list.pop_back();
        }

        log()->get(LogLevel::Debug2) << active_list.size() << " elements, " << samples.size() << " samples" << std::endl;
    }

    log()->get(LogLevel::Debug) << "Terminated with " << active_list.size() << " active samples remaining" << std::endl;
    log()->get(LogLevel::Debug) << "Resampled point cloud has " << samples.size() << " points" << std::endl;

    // append samples to output
    for (auto const& s : samples)
        output->appendPoint(*input, s);

    // if (cloud_f->points.empty())
    // {
    //     log()->get(LogLevel::Debug2) << "Filtered cloud has no points!" << std::endl;
    //     return viewSet;
    // }
    //
    // pclsupport::PCDtoPDAL(*cloud_f, output, bounds);
    //
    // log()->get(LogLevel::Debug2) << cloud->points.size() << " before, " <<
    //                              cloud_f->points.size() << " after" << std::endl;
    // log()->get(LogLevel::Debug2) << output->size() << std::endl;

    return viewSet;
}

} // namespace pdal
