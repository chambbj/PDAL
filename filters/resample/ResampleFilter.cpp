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

#include "ResampleFilter.hpp"

#include <pdal/util/Utils.hpp>

#include <cctype>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.resample", "Resample points with replacement.",
               "http://pdal.io/stages/filters.resample.html");

CREATE_STATIC_PLUGIN(1, 0, ResampleFilter, Filter, s_info)

std::string ResampleFilter::getName() const
{
    return s_info.name;
}


  Options ResampleFilter::getDefaultOptions()
  {
      Options options;

      options.add("mean", 0.0);
      options.add("stdev", 0.0);

      return options;
  }


void ResampleFilter::processOptions(const Options& options)
{
    m_mean = options.getValueOrDefault<double>("mean");
    m_stdev = options.getValueOrDefault<double>("stdev");
}


void ResampleFilter::addDimensions(PointLayoutPtr layout)
{
    m_index = layout->registerOrAssignDim("Index", Dimension::Type::Unsigned32);
}


PointViewSet ResampleFilter::run(PointViewPtr inView)
{
    PointViewSet viewSet;
    if (!inView->size())
        return viewSet;

    PointViewPtr outView = inView->makeNew();

    for (PointId i = 0; i < inView->size(); ++i)
    {
        // pick point at random in range [0,inView->size())
        std::random_device rd;
        PointId id = Utils::uniform(0, inView->size(), rd());

        // perturb point with given mean/stdev
        PointViewPtr tempView = inView->makeNew();
        tempView->appendPoint(*inView, id);
        double x = tempView->getFieldAs<double>(Dimension::Id::X, 0);
        double y = tempView->getFieldAs<double>(Dimension::Id::Y, 0);
        double z = tempView->getFieldAs<double>(Dimension::Id::Z, 0);
        tempView->setField(Dimension::Id::X, 0, x+Utils::normal(m_mean, m_stdev, rd()));
        tempView->setField(Dimension::Id::Y, 0, y+Utils::normal(m_mean, m_stdev, rd()));
        tempView->setField(Dimension::Id::Z, 0, z+Utils::normal(m_mean, m_stdev, rd()));
        tempView->setField(m_index, 0, id);
        outView->appendPoint(*tempView, 0);
    }

    viewSet.insert(outView);

    return viewSet;
}

/*
void ColorizationFilter::filter(PointView& view)
{
    std::vector<double> data;

    for (PointId idx = 0; idx < view.size(); ++idx)
    {
        double x = view.getFieldAs<double>(Dimension::Id::X, idx);
        double y = view.getFieldAs<double>(Dimension::Id::Y, idx);

        if (!m_raster->read(x, y, data))
            continue;

        int i(0);
        for (auto bi = m_bands.begin(); bi != m_bands.end(); ++bi)
        {
            BandInfo& b = *bi;
            view.setField(b.m_dim, idx, data[i] * b.m_scale);
            ++i;
        }
    }
}
*/

} // pdal
