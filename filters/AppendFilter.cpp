/******************************************************************************
 * Copyright (c) 2017, Bradley J Chambers (brad.chambers@gmail.com)
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

#include "AppendFilter.hpp"

#include <pdal/util/ProgramArgs.hpp>

namespace pdal
{

static PluginInfo const s_info {
	"filters.append",
	"Append dimensions.",
        "http://pdal.io/stages/filters.append.html"
};

CREATE_STATIC_STAGE(AppendFilter, s_info)

AppendFilter::AppendFilter()
{}

std::string AppendFilter::getName() const
{
    return s_info.name;
}

void AppendFilter::addArgs(ProgramArgs& args)
{
    args.add("filename", "Output filename", m_filename);
    args.add("dimension", "Dimension containing data to be appended", m_dimName);
}

void AppendFilter::addDimensions(PointLayoutPtr layout)
{
    m_dimId = layout->registerOrAssignDim(m_dimName, Dimension::Type::Double);
}

void AppendFilter::filter(PointView& inView)
{
    std::ifstream file;
    file.open(m_filename, std::ios::in | std::ios::binary);
    for (PointId idx = 0; idx < inView.size(); ++idx)
    {
        double val;
        file.read(reinterpret_cast<char*>(&val), sizeof(val));
        inView.setField(m_dimId, idx, val);
    }
}

} // pdal
