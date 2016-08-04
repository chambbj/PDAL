/******************************************************************************
* Copyright (c) 2016, Bradley J Chambers (brad.chambers@gmail.com)
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
#include <pdal/plugin.hpp>

#include <Eigen/Dense>

#include <memory>
#include <unordered_map>

extern "C" int32_t SMRFilter_ExitFunc();
extern "C" PF_ExitFunc SMRFilter_InitPlugin();

namespace pdal
{

using namespace Eigen;

class PointLayout;
class PointView;

class PDAL_DLL SMRFilter : public Filter
{
public:
    SMRFilter() : Filter()
    {}

    static void * create();
    static int32_t destroy(void *);
    std::string getName() const;

private:
    bool m_classify;
    bool m_extract;
    int m_numRows;
    int m_numCols;
    int m_maxRow;
    double m_cellSize;
    double m_percentSlope;
    double m_maxWindow;
    double m_threshold;
    BOX2D m_bounds;

    virtual void addArgs(ProgramArgs& args);
    virtual void addDimensions(PointLayoutPtr layout);
    int clamp(int t, int min, int max);
    int getColIndex(double x, double cell_size);
    int getRowIndex(double y, double cell_size);
    // MatrixXd inpaint(MatrixXd data);
    MatrixXd matrixOpen(MatrixXd data, int radius);
    MatrixXd createDSM(MatrixXd const& cx, MatrixXd const& cy, PointViewPtr view);
    MatrixXi progressiveFilter(MatrixXd const& ZImin, double cell_size, double slope, double max_window);
    // double interp2(int r, int c, MatrixXd cx, MatrixXd cy, MatrixXd cz);
    MatrixXd padMatrix(MatrixXd data, int radius);
    std::vector<PointId> processGround(PointViewPtr view);
    virtual PointViewSet run(PointViewPtr view);
    MatrixXd TPS(MatrixXd cx, MatrixXd cy, MatrixXd cz);
    void writeMatrix(MatrixXd data, std::string filename, double cell_size, PointViewPtr view);

    SMRFilter& operator=(const SMRFilter&); // not implemented
    SMRFilter(const SMRFilter&); // not implemented
};

} // namespace pdal
