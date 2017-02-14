/******************************************************************************
 * Copyright (c) 2017, Bradley J Chambers, brad.chambers@gmail.com
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

#include "ElevationResidualAnalysisWriter.hpp"

#include <pdal/EigenUtils.hpp>
#include <pdal/PointView.hpp>
#include <pdal/util/Utils.hpp>

namespace pdal
{
static PluginInfo const s_info{"writers.era",
                               "ElevationResidualAnalysis writer",
                               "http://pdal.io/stages/writers.era.html"};

CREATE_STATIC_STAGE(ElevationResidualAnalysisWriter, s_info)

std::string ElevationResidualAnalysisWriter::getName() const
{
    return s_info.name;
}

ElevationResidualAnalysisWriter::ElevationResidualAnalysisWriter() : Writer()
{
}

void ElevationResidualAnalysisWriter::addArgs(ProgramArgs& args)
{
    args.add("filename", "Output filename", m_filename).setPositional();
    args.add("edge_length", "Edge length", m_edgeLength, 15.0);
    args.add("primitive_type", "Primitive type", m_primTypesSpec, {"mean"});
    args.add("driver", "GDAL format driver", m_driver, "GTiff");
}

void ElevationResidualAnalysisWriter::initialize()
{
    static std::map<std::string, PrimitiveType> primtypes = {
        {"mean", MEAN}, {"diff_mean", DIFF_MEAN}, {"range", RANGE}};

    auto hashPos = handleFilenameTemplate(m_filename);
    if (hashPos == std::string::npos && m_primTypesSpec.size() > 1)
        throwError("No template placeholder ('#') found in filename '" +
                   m_filename +
                   "' when one is required with multiple primitive "
                   "types.");

    for (std::string os : m_primTypesSpec)
    {
        std::string s = Utils::tolower(os);
        auto pi = primtypes.find(s);
        if (pi == primtypes.end())
            throwError("Unrecognized primitive type '" + os + "'.");
        TypeOutput to;
        to.m_type = pi->second;
        to.m_filename = generateFilename(pi->first, hashPos);
        m_primitiveTypes.push_back(to);
    }
}

std::string ElevationResidualAnalysisWriter::generateFilename(
    const std::string& primName, std::string::size_type hashPos) const
{
    std::string filename = m_filename;
    if (hashPos != std::string::npos)
        filename.replace(hashPos, 1, primName);
    return filename;
}

void ElevationResidualAnalysisWriter::write(const PointViewPtr data)
{
    using namespace Eigen;

    // Bounds are required for computing number of rows and columns, and for
    // later indexing individual points into the appropriate raster cells.
    BOX2D bounds;
    data->calculateBounds(bounds);
    SpatialReference srs = data->spatialReference();

    // Determine the number of rows and columns at the given cell size.
    size_t cols = ((bounds.maxx - bounds.minx) / m_edgeLength) + 1;
    size_t rows = ((bounds.maxy - bounds.miny) / m_edgeLength) + 1;

    // Begin by creating a DSM of mean elevations per XY cell.
    MatrixXd meanMatrix =
        createMeanMatrix(*data.get(), rows, cols, m_edgeLength, bounds);
    MatrixXd maxMatrix =
        createMaxMatrix2(*data.get(), rows, cols, m_edgeLength, bounds);
    MatrixXd minMatrix =
        createMinMatrix2(*data.get(), rows, cols, m_edgeLength, bounds);

    writeMatrix(meanMatrix, m_primitiveTypes[0].m_filename, m_driver,
                m_edgeLength, bounds, srs);
    writeMatrix(maxMatrix - minMatrix, m_primitiveTypes[1].m_filename, m_driver,
                m_edgeLength, bounds, srs);
}

} // namespace pdal
