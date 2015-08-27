/******************************************************************************
* Copyright (c) 2015, Bradley J Chambers, brad.chambers@gmail.com
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

#include "DerivativeWriter.hpp"

#include <sstream>
#include <string>

#include <pdal/PointView.hpp>
#include <pdal/Raster.hpp>

namespace pdal
{
static PluginInfo const s_info =
    PluginInfo("writers.derivative", "Derivative writer",
               "http://pdal.io/stages/writers.derivative.html");

CREATE_STATIC_PLUGIN(1, 0, DerivativeWriter, Writer, s_info)

std::string DerivativeWriter::getName() const
{
    return s_info.name;
}

const float c_background = FLT_MIN;

DerivativeWriter::DerivativeWriter()
    : Writer()
    , m_primitive_type(0)
{}


void DerivativeWriter::ready(PointTableRef table)
{
    m_inSRS = table.spatialRef();
}

void DerivativeWriter::initialize()
{
    m_spacing_x = getOptions().getValueOrDefault<double>("x_spacing", 15.0);
    m_spacing_y = getOptions().getValueOrDefault<double>("y_spacing", 15.0);
    m_filename = getOptions().getValueOrThrow<std::string>("filename");

    std::string primitive_type =
        getOptions().getValueOrDefault<std::string>("primitive_type", "slope_d8");

    if (Utils::iequals(primitive_type, "slope_d8"))
        m_primitive_type = Utils::SLOPE_D8;
    else if (Utils::iequals(primitive_type, "slope_fd"))
        m_primitive_type = Utils::SLOPE_FD;
    else if (Utils::iequals(primitive_type, "aspect_d8"))
        m_primitive_type = Utils::ASPECT_D8;
    else if (Utils::iequals(primitive_type, "aspect_fd"))
        m_primitive_type = Utils::ASPECT_FD;
    else if (Utils::iequals(primitive_type, "hillshade"))
        m_primitive_type = Utils::HILLSHADE;
    else if (Utils::iequals(primitive_type, "contour_curvature"))
        m_primitive_type = Utils::CONTOUR_CURVATURE;
    else if (Utils::iequals(primitive_type, "profile_curvature"))
        m_primitive_type = Utils::PROFILE_CURVATURE;
    else if (Utils::iequals(primitive_type, "tangential_curvature"))
        m_primitive_type = Utils::TANGENTIAL_CURVATURE;
    else if (Utils::iequals(primitive_type, "total_curvature"))
        m_primitive_type = Utils::TOTAL_CURVATURE;
    else if (Utils::iequals(primitive_type, "catchment_area"))
        m_primitive_type = Utils::CATCHMENT_AREA;
    else
    {
        std::ostringstream oss;
        oss << "Unrecognized primitive type " << primitive_type;
        throw pdal_error(oss.str().c_str());
    }
}


Options DerivativeWriter::getDefaultOptions()
{
    Options options;

    options.add("x_spacing", 15.0, "X post spacing");
    options.add("y_spacing", 15.0, "Y post spacing");
    options.add("primitive_type", "slope_d8", "Primitive type");

    return options;
}


void DerivativeWriter::write(const PointViewPtr data)
{
    Utils::Raster foo(data, m_spacing_x, m_spacing_y, m_inSRS, c_background, log());

    switch (m_primitive_type)
    {
        case Utils::SLOPE_D8:
            foo.writeSlope(m_filename, Utils::SD8);
            break;

        case Utils::SLOPE_FD:
            foo.writeSlope(m_filename, Utils::SFD);
            break;

        case Utils::ASPECT_D8:
            foo.writeAspect(m_filename, Utils::AD8);
            break;

        case Utils::ASPECT_FD:
            foo.writeAspect(m_filename, Utils::AFD);
            break;

        case Utils::HILLSHADE:
            foo.writeHillshade(m_filename);
            break;

        case Utils::CONTOUR_CURVATURE:
            foo.writeCurvature(m_filename, Utils::CONTOUR, c_background);
            break;

        case Utils::PROFILE_CURVATURE:
            foo.writeCurvature(m_filename, Utils::PROFILE, c_background);
            break;

        case Utils::TANGENTIAL_CURVATURE:
            foo.writeCurvature(m_filename, Utils::TANGENTIAL, c_background);
            break;

        case Utils::TOTAL_CURVATURE:
            foo.writeCurvature(m_filename, Utils::TOTAL, c_background);
            break;

        case Utils::CATCHMENT_AREA:
            foo.writeCatchmentArea(m_filename);
            break;
    }
}
} // namespace pdal
