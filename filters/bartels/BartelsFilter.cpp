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

#include "BartelsFilter.hpp"

#include <pdal/KDIndex.hpp>
#include <pdal/pdal_macros.hpp>
#include <pdal/util/ProgramArgs.hpp>

namespace pdal
{

static PluginInfo const s_info =
    PluginInfo("filters.bartels", "Bartels & Wei Skewness Balancing",
               "http://pdal.io/stages/filters.bartels.html");

CREATE_STATIC_PLUGIN(1, 0, BartelsFilter, Filter, s_info)

std::string BartelsFilter::getName() const
{
    return s_info.name;
}


void BartelsFilter::addArgs(ProgramArgs& args)
{
    args.add("classify", "Apply classification labels?", m_classify, true);
    args.add("extract", "Extract ground returns?", m_extract);
}


void BartelsFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Dimension::Id::Classification);
}

std::set<PointId> BartelsFilter::processGround(PointViewPtr view)
{
    point_count_t np(view->size());

    std::set<PointId> groundIdx;
    for (PointId i = 0; i < np; ++i)
        groundIdx.insert(i);
        
    double skewness;

    do
    {
        // std::cerr << "Computing skewness for " << groundIdx.size() << " samples\n";
        PointId apex;
        point_count_t n(0);
        point_count_t n1(0);
        double delta, delta_n, term1, M1, M2, M3;
        M1 = M2 = M3 = 0.0;
        double maxz = std::numeric_limits<double>::lowest();

        // compute initial skewness, locate apex, and seed ground indices
        for (auto const& i : groundIdx)
        {
            double z = view->getFieldAs<double>(Dimension::Id::Z, i);
            if (z > maxz)
            {
                apex = i;
                maxz = z;
            }

            n1 = n;
            n++;
            delta = z - M1;
            delta_n = delta / n;
            term1 = delta * delta_n * n1;
            M1 += delta_n;
            M3 += term1 * delta_n * (n-2)-3 * delta_n * M2;
            M2 += term1;
        }

        skewness = std::sqrt(n) * M3 / std::pow(M2, 1.5);
        // std::cerr << "Mean is " << M1 << std::endl;
        // std::cerr << "Skewness is " << skewness << std::endl;
        if (skewness > 0)
            groundIdx.erase(apex);
    }
    while (skewness > 0);

    return groundIdx;
}

PointViewSet BartelsFilter::run(PointViewPtr input)
{
    bool logOutput = log()->getLevel() > LogLevel::Debug1;
    if (logOutput)
        log()->floatPrecision(8);
    log()->get(LogLevel::Debug2) << "Process BartelsFilter...\n";

    auto idx = processGround(input);

    PointViewSet viewSet;
    if (!idx.empty() && (m_classify || m_extract))
    {

        if (m_classify)
        {
            log()->get(LogLevel::Debug2) << "Labeled " << idx.size() << " ground returns!\n";

            // set the classification label of ground returns as 2
            // (corresponding to ASPRS LAS specification)
            for (const auto& i : idx)
            {
                input->setField(Dimension::Id::Classification, i, 2);
            }

            viewSet.insert(input);
        }

        if (m_extract)
        {
            log()->get(LogLevel::Debug2) << "Extracted " << idx.size() << " ground returns!\n";

            // create new PointView containing only ground returns
            PointViewPtr output = input->makeNew();
            for (const auto& i : idx)
            {
                output->appendPoint(*input, i);
            }

            viewSet.erase(input);
            viewSet.insert(output);
        }
    }
    else
    {
        if (idx.empty())
            log()->get(LogLevel::Debug2) << "Filtered cloud has no ground returns!\n";

        if (!(m_classify || m_extract))
            log()->get(LogLevel::Debug2) << "Must choose --classify or --extract\n";

        // return the input buffer unchanged
        viewSet.insert(input);
    }

    return viewSet;
}

} // namespace pdal
