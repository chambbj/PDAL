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

#include "MorphologicalErodeFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/util/ProgramArgs.hpp>

namespace pdal
{
using namespace Dimension;

static StaticPluginInfo const s_info{
    "filters.morpherode", "Morphological erosion",
    "http://pdal.io/stages/filters.morpherode.html"};

CREATE_STATIC_STAGE(MorphologicalErodeFilter, s_info)

struct MorphologicalErodeArgs
{
    double m_window;
    //TODO(chambbj): allow flexibility in the dimension to be opened!
};

MorphologicalErodeFilter::MorphologicalErodeFilter() : m_args(new MorphologicalErodeArgs) {}

MorphologicalErodeFilter::~MorphologicalErodeFilter() {}

std::string MorphologicalErodeFilter::getName() const
{
    return s_info.name;
}

void MorphologicalErodeFilter::addArgs(ProgramArgs& args)
{
    args.add("window", "Window size", m_args->m_window, 11.0);
}

void MorphologicalErodeFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::Erosion);
}

void MorphologicalErodeFilter::filter(PointView& view)
{
    if (!view.size())
        return;

    const KD2Index& kdi = view.build2dIndex();

    // erode, find min per Z
    for (PointRef p : view)
    {
	    PointIdList ids = kdi.radius(p, m_args->m_window);
	    std::vector<double> z(ids.size());
	    for (size_t i = 0; i < ids.size(); ++i)
		    z[i] = view.getFieldAs<double>(Id::Z, ids[i]);
	    auto it = std::min_element(z.begin(), z.end());
	    p.setField(Id::Erosion, *it);
    }
}

} // namespace pdal
