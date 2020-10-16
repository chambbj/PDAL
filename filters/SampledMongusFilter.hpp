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

#pragma once

#include <pdal/Filter.hpp>

#include <string>

namespace pdal
{

class Options;

class PDAL_DLL SampledMongusFilter : public pdal::Filter
{
public:
    SampledMongusFilter() : Filter() {}
    SampledMongusFilter& operator=(const SampledMongusFilter&) = delete;
    SampledMongusFilter(const SampledMongusFilter&) = delete;

    std::string getName() const;

private:
    double m_radius;
    int m_count;
    int m_maxiters;
    double m_thresh;
    double m_radDecay;
    double m_threshDecay;
    double m_lambda;
    double m_lambdaDecay;
    Arg* m_lambdaArg;
    BOX3D m_bounds;
    double m_maxrange;

    virtual void addArgs(ProgramArgs& args);
    virtual void addDimensions(PointLayoutPtr layout);
    virtual void filter(PointView& view);

    PointIdList sample(PointView& view);
    PointIdList sample(PointView& view, PointIdList ids);
    PointIdList foo(PointView& view, PointIdList ids);
    void bar(PointView& view, PointIdList ids);
    void baz(PointView& view, PointIdList ids);
    void interpolate(PointViewPtr candView, PointViewPtr gView);
    void interpolate(PointView& view, PointIdList ids, PointViewPtr gView);
    void tophat(PointViewPtr candView);
    void tophat(PointView& view, PointIdList ids);
    std::vector<PointIdList> buildScaleSpace(PointView& view);
};

} // namespace pdal
