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

#include <Eigen/Dense>

#include <pdal/pdal_types.hpp>
#include <pdal/Dimension.hpp>
#include <pdal/PointView.hpp>

#include <vector>

namespace pdal
{

using namespace Eigen;

class ClusterNode
{
public:
    bool m_coplanar;

    ClusterNode(PointViewPtr view, PointIdList ids);

    double area()
    {
        return m_xEdge * m_yEdge * m_zEdge;
    }

    // Initialize node by computing the centroid, covariance, and eigen
    // decomposition of the node samples.
    void initialize();

    std::vector<PointIdList> children();

    // Accessors

    Vector3d centroid()
    {
        return m_centroid;
    }

    Matrix<double, 3, 1, 0, 3, 1> eigenvalues()
    {
        return m_eigenvalues;
    }

    PointIdList indices()
    {
        return m_ids;
    }

    Matrix<double, 3, 1, 0, 3, 1> normal()
    {
        return m_normal;
    }

    void refineFit();

    Matrix3d xyzCovariance()
    {
        return m_covariance;
    }

    point_count_t size()
    {
        return m_ids.size();
    }

    Matrix3d computeJacobian();

    void compute();

    void vote(double totalArea, point_count_t totalPoints);

private:
    BOX3D m_bounds;
    Vector3d m_centroid;
    Matrix3d m_covariance;
    Matrix3d m_Jacobian;
    Matrix3d m_polarCov;
    Matrix<double, 3, 1, 0, 3, 1> m_eigenvalues;
    Matrix3d m_eigenvectors;
    PointIdList m_ids, m_originalIds;
    Matrix<double, 3, 1, 0, 3, 1> m_normal;
    PointViewPtr m_view;
    double m_xEdge, m_yEdge, m_zEdge;
    Vector3d m_gmin;
};

} // namespace pdal
