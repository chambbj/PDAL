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

#include <pdal/EigenUtils.hpp>

#include <Eigen/Dense>

namespace pdal
{

using namespace Dimension;
using namespace Eigen;

const double c_PI = std::acos(-1.0);

namespace
{

class Node
{
public:
    Node(PointViewPtr view, PointIdList ids)
    {
        m_view = view;
        m_ids = ids;
        m_originalIds = m_ids;

        // Compute the bounds of the current node. (Scoped so that we will let
        // go of the PointViewPtr.)
        {
            // Create a temporary PointView for the current node.
            PointViewPtr node = m_view->makeNew();
            for (PointId const& i : m_ids)
                node->appendPoint(*m_view, i);

            node->calculateBounds(m_bounds);
        }

        // Compute some stats on the current node.
        m_xEdge = (m_bounds.maxx - m_bounds.minx);
        m_yEdge = (m_bounds.maxy - m_bounds.miny);
        m_zEdge = (m_bounds.maxz - m_bounds.minz);
        // std::cerr << m_bounds << std::endl;
        // std::cerr << "Constructing node with " << m_xEdge << ", " << m_yEdge
        // << ", " << m_zEdge << std::endl;
    }

    double area()
    {
        return m_xEdge * m_yEdge * m_zEdge;
    }

    // Initialize node by computing the centroid, covariance, and eigen
    // decomposition of the node samples.
    void initialize()
    {
        // std::cerr << m_ids.size() << std::endl;
        m_centroid = computeCentroid(*m_view, m_ids);
        m_covariance = computeCovariance(*m_view, m_centroid, m_ids);
        SelfAdjointEigenSolver<Matrix3d> solver;
        solver.compute(m_covariance);
        if (solver.info() != Success)
            throw pdal_error("Cannot perform eigen decomposition.");
        m_eigenvalues = solver.eigenvalues();
        m_eigenvectors = solver.eigenvectors();
        m_normal = m_eigenvectors.col(0);
        // std::cerr << "Initializing node\n";
    }

    std::vector<PointIdList> children()
    {
        double maxEdge = std::max(m_xEdge, std::max(m_yEdge, m_zEdge));
        double xSplit = maxEdge / 2 + m_bounds.minx;
        double ySplit = maxEdge / 2 + m_bounds.miny;
        double zSplit = maxEdge / 2 + m_bounds.minz;
        // double xSplit = m_xEdge/2 + m_bounds.minx;
        // double ySplit = m_yEdge/2 + m_bounds.miny;
        // double zSplit = m_zEdge/2 + m_bounds.minz;

        // Create eight vectors of points indices for the eight children of the
        // current node.
        PointIdList upNW, upNE, upSE, upSW, downNW, downNE, downSE, downSW;

        // Populate the children, using original_ids as we may have removed
        // points from the ids vector during refinement.
        for (PointId const& i : m_originalIds)
        {
            double x = m_view->getFieldAs<double>(Id::X, i);
            double y = m_view->getFieldAs<double>(Id::Y, i);
            double z = m_view->getFieldAs<double>(Id::Z, i);

            if (x < xSplit && y >= ySplit && z >= zSplit)
                upNW.push_back(i);
            if (x >= xSplit && y >= ySplit && z >= zSplit)
                upNE.push_back(i);
            if (x >= xSplit && y < ySplit && z >= zSplit)
                upSE.push_back(i);
            if (x < xSplit && y < ySplit && z >= zSplit)
                upSW.push_back(i);
            if (x < xSplit && y >= ySplit && z < zSplit)
                downNW.push_back(i);
            if (x >= xSplit && y >= ySplit && z < zSplit)
                downNE.push_back(i);
            if (x >= xSplit && y < ySplit && z < zSplit)
                downSE.push_back(i);
            if (x < xSplit && y < ySplit && z < zSplit)
                downSW.push_back(i);
        }

        std::vector<PointIdList> children{upNW,   upNE,   upSE,   upSW,
                                          downNW, downNE, downSE, downSW};

        return children;
    }

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

    void refineFit()
    {
        double threshold = std::max(m_xEdge, std::max(m_yEdge, m_zEdge)) / 10;

        // Create a plane with given normal (equal to eigenvector corresponding
        // to smallest eigenvalue) and passing through the centroid.
        Hyperplane<double, 3> plane = Hyperplane<double, 3>(m_normal, 0);

        // Iterate over node indices and only keep those that are within a given
        // tolerance of the plane surface.
        PointIdList new_ids;
        for (PointId const& j : m_ids)
        {
            PointRef p(m_view->point(j));
            Vector3d pt(p.getFieldAs<double>(Id::X),
                        p.getFieldAs<double>(Id::Y),
                        p.getFieldAs<double>(Id::Z));
            // Vector3d pt = pointRefToVector3f(m_view->point(j));
            pt = pt - m_centroid;
            if (plane.absDistance(pt) < threshold)
                new_ids.push_back(j);
        }
        // std::cerr << "Threshold computed as " << threshold << std::endl;
        // if (new_ids.size() != m_ids.size())
        //     std::cerr << "Eureka! " << new_ids.size() << " != " <<
        //     m_ids.size() << std::endl;
        m_ids.swap(new_ids);

        initialize();
    }

    // Compute the Jacobian and convert covariance in XYZ space to covariance in
    // polar coordinates.
    Matrix3d polarCovariance()
    {
        Matrix3d J = computeJacobian();
        return J * m_covariance * J.transpose();
    }

    point_count_t size()
    {
        return m_ids.size();
    }

private:
    BOX3D m_bounds;
    Vector3d m_centroid;
    Matrix3d m_covariance;
    Matrix<double, 3, 1, 0, 3, 1> m_eigenvalues;
    Matrix3d m_eigenvectors;
    PointIdList m_ids, m_originalIds;
    Matrix<double, 3, 1, 0, 3, 1> m_normal;
    PointViewPtr m_view;
    double m_xEdge, m_yEdge, m_zEdge;

    Matrix3d computeJacobian()
    {
        double rho = m_centroid.dot(m_normal);
        double rho2 = rho * rho;
        Vector3d p = rho * m_normal;
        double angle = std::acos(rho / (m_centroid.norm() * m_normal.norm()));
        // std::cerr << "Angle is " << angle*180/c_PI << std::endl;
        if (angle * 180 / c_PI > 90)
        {
            m_normal *= -1;
            // std::cerr << "Flipping normal to " << normal.transpose() <<
            // std::endl;
            rho = m_centroid.dot(m_normal);
            rho2 = rho * rho;
            p = rho * m_normal;
        }
        double w = p.x() * p.x() + p.y() * p.y();
        double rootw = std::sqrt(w);
        Matrix3d J;
        J.row(0) << m_normal.x(), m_normal.y(), m_normal.z();
        J.row(1) << p.x() * p.z() / (rootw * rho2),
            p.y() * p.z() / (rootw * rho2), -rootw / rho2;
        J.row(2) << -p.y() / w, p.x() / w, 0;

        return J;
    }
};

} // namespace

class PDAL_DLL KHTFilter : public Filter
{
public:
    KHTFilter();
    ~KHTFilter();
    KHTFilter& operator=(const KHTFilter&) = delete;
    KHTFilter(const KHTFilter&) = delete;

    std::string getName() const;

private:
    double m_totalArea;
    point_count_t m_totalPoints;

    virtual void addDimensions(PointLayoutPtr layout);
    void cluster(PointViewPtr view, std::vector<PointId> ids, int level,
                 std::deque<Node>& nodes);
    void foo();
    virtual PointViewSet run(PointViewPtr view);
};

} // namespace pdal
