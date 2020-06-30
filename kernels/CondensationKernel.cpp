/******************************************************************************
* Copyright (c) 2015, Hobu Inc. (info@hobu.co)
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

#include "CondensationKernel.hpp"

#include <filters/CondensationFilter.hpp>
#include <pdal/StageFactory.hpp>

namespace pdal
{

static PluginInfo const s_info
{
    "kernels.condensation",
    "Condensation Kernel",
    "http://pdal.io/kernels/kernels.condensation.html"
};

CREATE_STATIC_STAGE(CondensationKernel, s_info)

std::string CondensationKernel::getName() const
{
    return s_info.name;
}


void CondensationKernel::addSwitches(ProgramArgs& args)
{
    args.add("files,f", "input/output files", m_files).setPositional();
    args.add("xmin", "Minimum X", m_xmin);
    args.add("xmax", "Maximum X", m_xmax);
    args.add("ymin", "Minimum Y", m_ymin);
    args.add("ymax", "Maximum Y", m_ymax);
    args.add("zmin", "Minimum Z", m_zmin);
    args.add("zmax", "Maximum Z", m_zmax);
    args.add("nsamps", "Number of samples", m_nSamples);
}


void CondensationKernel::validateSwitches(ProgramArgs& args)
{
    if (m_files.size() < 2)
        throw pdal_error("Must specify an input and output file.");
    m_outputFile = m_files.back();
    m_files.resize(m_files.size() - 1);
}


int CondensationKernel::execute()
{
    PointTable table;
  printf("Set %0.2f\t%0.2f\n", m_ymin, m_ymax);  
    Options filterOptions;
    filterOptions.add("xmin", m_xmin);
    filterOptions.add("xmax", m_xmax);
    filterOptions.add("ymin", m_ymin);
    filterOptions.add("ymax", m_ymax);
    filterOptions.add("zmin", m_zmin);
    filterOptions.add("zmax", m_zmax);
    filterOptions.add("nsamps", m_nSamples);
    filterOptions.add("verbose", 5);
    filterOptions.add("debug", true);

    CondensationFilter filter;
    filter.addOptions(filterOptions);

    for (size_t i = 0; i < m_files.size(); ++i)
    {
        Stage& reader = makeReader(m_files[i], "");
        filter.setInput(reader);
    }

    Stage& writer = makeWriter(m_outputFile, filter, "");
    writer.prepare(table);
    writer.execute(table);
    return 0;
}

} // namespace pdal
