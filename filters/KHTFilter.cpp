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

#include "KHTFilter.hpp"

#include <Eigen/Dense>

#include <numeric>

namespace pdal
{

using namespace Dimension;
using namespace Eigen;

static PluginInfo const s_info{"filters.kht", "3D-KHT filter",
                               "http://pdal.io/stages/filters.kht.html"};

CREATE_STATIC_STAGE(KHTFilter, s_info)

KHTFilter::KHTFilter() {}

KHTFilter::~KHTFilter() {}

std::string KHTFilter::getName() const
{
    return s_info.name;
}

void KHTFilter::addDimensions(PointLayoutPtr layout)
{
    layout->registerDim(Id::Classification);
}

/*
Algorithm Summary

1. Generate octree
2. For each node
  a. If ns < 30, return, node is not coplanar
  b. If nl > 4, check for coplanar points
    i. Compute covariance in XYZ
    ii. Test eigenvalues for coplanarity
    iii. Perform plane fitting and voting here?
  c. Recursively check child nodes
  d. If coplanar
    0. Perform plane fitting here?
    i. Transform into spherical coordinates
    ii. Compute covariance in theta, phi, rho
    iii. Vote and update accumulator
3. Filter accumulator
4. Sort accumulator
5. Detect peaks in accumulator
*/

void KHTFilter::foo()
{
    /*
    PointIdList original_ids(ids);
            n.refineFit();

            Matrix3d Sigma = n.polarCovariance();
            Sigma(0, 0) += 0.001; // to avoid singular cases
            // log()->get(LogLevel::Debug) << "Sigma: " << Sigma << std::endl;

            // Perform the eigen decomposition with the covariance in polar
            // coordinates.
            SelfAdjointEigenSolver<Matrix3d> solver;
            solver.compute(Sigma);
            if (solver.info() != Success)
                throw pdal_error("Cannot perform eigen decomposition.");
            double stdev = std::sqrt(solver.eigenvalues()[0]);
            MatrixXd gmin = 2 * stdev * solver.eigenvectors().col(0);

            Matrix3d SigmaInv;
            double SigmaDet;
            bool invertible;
            Sigma.computeInverseAndDetWithCheck(SigmaInv, SigmaDet, invertible);

            if (!invertible)
            {
                log()->get(LogLevel::Debug) << "sigma is not invertible\n";
                return;
            }

            // log()->get(LogLevel::Debug) << "Determinant: " << SigmaDet <<
    std::endl;
            // log()->get(LogLevel::Debug) << "Inverse: " << SigmaInv <<
    std::endl;

            double factor = 1 / (15.7496 * std::sqrt(SigmaDet));
            // log()->get(LogLevel::Debug) << "Factor: " << factor << std::endl;

            double weight = 0.75 * (n.area() / m_totalArea) +
                            0.25 * (n.size() / m_totalPoints);
            // log()->get(LogLevel::Debug) << "Weight: " << weight << std::endl;

            // compute centroid of polar coordinates
            double rhoSum = 0.0;
            double thetaSum = 0.0;
            double phiSum = 0.0;
            for (PointId const& j : n.indices())
            {
                Vector3d pt(view->getFieldAs<double>(Id::X, j),
                            view->getFieldAs<double>(Id::Y, j),
                            view->getFieldAs<double>(Id::Z, j));
                pt = pt - n.centroid();
                double rho = std::sqrt(pt.x() * pt.x() + pt.y() * pt.y() +
                                       pt.z() * pt.z());
                rhoSum += rho;
                thetaSum += std::atan(pt.y() / pt.x());
                phiSum += std::acos(pt.z() / rho);
            }
            Vector3d polarCentroid(rhoSum / n.size(), phiSum / n.size(),
                                   thetaSum / ids.size());

            for (PointId const& j : n.indices())
            {
                Vector3d pt(view->getFieldAs<double>(Id::X, j),
                            view->getFieldAs<double>(Id::Y, j),
                            view->getFieldAs<double>(Id::Z, j));
                pt = pt - n.centroid();
                double rho = std::sqrt(pt.x() * pt.x() + pt.y() * pt.y() +
                                       pt.z() * pt.z());
                double theta = std::atan(pt.y() / pt.x());
                double phi = std::acos(pt.z() / rho);
                Vector3d q(rho, phi, theta);
                q = q - polarCentroid;
                double temp = -0.5 * q.transpose() * SigmaInv * q;
                double temp2 = factor * std::exp(temp);
                double temp3 = weight * temp2;
                // log()->get(LogLevel::Debug) << q.transpose() << std::endl;
                // log()->get(LogLevel::Debug) << std::exp(temp) << std::endl;
                // printf("%f.8\n", temp3);
                // printf("Bin %.2f %.2f %.2f gets a vote of %.8f\n",
                // theta*180/c_PI, phi*180/c_PI, rho, temp3);
            }

            // log()->get(LogLevel::Debug) << gmin << std::endl;

            log()->get(LogLevel::Debug2)
                << "Level: " << level << "\t"
                << "Points before: " << original_ids.size() << "\t"
                << "Points after: " << n.size() << "\t"
                << "Stdev: " << stdev << "\t"
                << "Area: " << n.area() << "\t"
                << "Eigenvalues: " << eigVal.transpose() << "\t" << std::endl;
            // throw pdal_error("foo");
            return;
    */
}

void KHTFilter::cluster(PointViewPtr view, PointIdList ids, int level,
                        std::deque<Node>& nodes)
{
    // If there are too few points in the current node, then bail. We need a
    // minimum number of points to determine whether or not the points are
    // coplanar and if this cluster can be used for voting.
    if (ids.size() < 30)
        return;

    bool coplanar(false);

    Node n(view, ids);

    if (level == 0)
    {
        m_totalArea = n.area();
        m_totalPoints = n.size();
    }

    // Don't even begin looking for clusters of coplanar points until we reach a
    // given level in the octree.
    if (level > 3)
    {
        n.initialize();
        Vector3d eigVal = n.eigenvalues();

        // right location?
        nodes.push_back(n);

        // Test plane thickness and isotropy.
        if ((eigVal[1] > 25 * eigVal[0]) && (6 * eigVal[1] > eigVal[2]))
        {
            coplanar = true;
            log()->get(LogLevel::Debug)
                << "Approx. coplanar node #" << nodes.size() << ": "
                << ids.size() << " points at level " << level << std::endl;
            return;
        }
        log()->get(LogLevel::Debug)
            << "Non-coplanar node #" << nodes.size() << ": " << ids.size()
            << " points at level " << level << std::endl;
    }

    // Loop over children and recursively cluster at the next level in the
    // octree.
    for (PointIdList const& child : n.children())
        cluster(view, child, level + 1, nodes);

    return;
}

PointViewSet KHTFilter::run(PointViewPtr view)
{
    // Quick check that we have any points to process
    PointViewSet viewSet;
    if (!view->size())
        return viewSet;

    // Gather initial set of PointIds for the PointView
    PointIdList ids(view->size());
    std::iota(ids.begin(), ids.end(), 0);

    // Begin hierarchical clustering at level 0 using all PointIds
    std::deque<Node> nodes;
    cluster(view, ids, 0, nodes);

    // Insert the clustered view and return
    viewSet.insert(view);
    return viewSet;
}

} // namespace pdal
